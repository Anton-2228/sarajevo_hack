"""The control plane, as a Python object.

Two things are worth knowing before reading the code.

*There is no `close_round`.* The round lifecycle belongs to the server's
operator, and this agent must never end one. That is enforced by not having the
capability rather than by remembering not to use it.

*Submit is the only call worth thinking about twice.* A resubmission into an
open round replaces the previous one and bumps its revision, so repeating it is
safe -- but it can be megabytes, so a lost response is resolved by asking
`/tasks` what happened rather than by re-sending on spec.

There is no authentication: the contract has no enrollment token, no node
secret and no request signing (CONTRACT.md 2.1). Nothing here sends a
credential because there is none to send.

*The node reads its corpus through `/shards/…`.* Contract 0.7.0 made the data
server-held, so fetching it is a lifecycle call like any other -- with one trap
worth naming here: an unknown shard answers 200 and an empty list, not 404.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

import requests

from node.agent import __version__
from node.agent.models import (
    HandshakeRequest,
    HandshakeResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    TaskView,
)

LOG = logging.getLogger("node.agent.api")

RETRY_STATUSES = {429, 502, 503, 504}
BACKOFF_BASE_S = 0.5
BACKOFF_CAP_S = 30.0


class ApiError(Exception):
    def __init__(
        self, message: str, *, status: int = 0, detail: str = "", body: Any = None, url: str = ""
    ) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail
        self.body = body
        self.url = url


class TransientError(ApiError):
    """Worth trying again: a timeout, a dropped connection, a 503."""


class UnknownNodeError(ApiError):
    """The server has no record of this node. Re-enrolment is the only fix."""


class RoundClosed(ApiError):
    """409 from ack or submit: the operator closed the round.

    There is no reopening, so the work is lost and retrying cannot help.
    """


class ValidationRejected(ApiError):
    """422. Carries the parsed FastAPI detail list, which usually explains it."""

    @property
    def fields(self) -> list[str]:
        """Names the server complained about, for the self-healing retry."""
        names: list[str] = []
        if isinstance(self.body, dict):
            for item in self.body.get("detail") or []:
                if isinstance(item, dict):
                    names.extend(str(p) for p in item.get("loc", []) if isinstance(p, str))
        return names

    @property
    def missing_metrics(self) -> list[str]:
        """Metric keys the round demanded and the payload did not carry.

        The server reports these as `type: missing_metrics` with the names in
        the message rather than in `loc`, so they are parsed out of the text.
        """
        import ast

        if not isinstance(self.body, dict):
            return []
        for item in self.body.get("detail") or []:
            if not isinstance(item, dict) or item.get("type") != "missing_metrics":
                continue
            message = str(item.get("msg", ""))
            start = message.find("[")
            if start < 0:
                continue
            try:
                parsed = ast.literal_eval(message[start:])
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(name) for name in parsed]
        return []


class ControlPlaneClient:
    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers["User-Agent"] = f"node-agent/{__version__}"

    def clone(self) -> ControlPlaneClient:
        """A second client for another thread.

        `requests.Session` is not documented thread-safe, and the heartbeat
        thread runs alongside a submit that can take minutes.
        """
        return ControlPlaneClient(
            self.base_url,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            max_retries=self.max_retries,
        )

    # -- endpoints --------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_nodes(self) -> list[dict[str, Any]]:
        return self._request("GET", "/nodes")

    def handshake(self, request: HandshakeRequest) -> HandshakeResponse:
        return HandshakeResponse.from_dict(
            self._request("POST", "/nodes/handshake", body=request.to_dict())
        )

    def heartbeat(self, node_id: str, beat: HeartbeatRequest) -> HeartbeatResponse:
        return HeartbeatResponse.from_dict(
            self._request("POST", f"/nodes/{node_id}/heartbeat", body=beat.to_dict())
        )

    def get_tasks(self, node_id: str) -> list[TaskView]:
        payload = self._request("GET", f"/tasks/{node_id}")
        return [TaskView.from_dict(t) for t in payload.get("tasks", [])]

    def ack(self, node_id: str, round_id: str) -> TaskView:
        # Documented idempotent, so a retry after a lost response is free.
        return TaskView.from_dict(
            self._request("POST", f"/tasks/{node_id}/{round_id}/ack")
        )

    def submit(self, node_id: str, round_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Send the scores. Resolves an ambiguous outcome by asking, not resending."""
        try:
            return self._request(
                "POST", f"/tasks/{node_id}/{round_id}/submit", body=body, retries=0
            )
        except TransientError as error:
            # The response never arrived. Resending would be correct -- an open
            # round accepts a replacement -- but the body can be megabytes, so
            # find out before spending them again.
            LOG.warning("submit response lost for round %s: %s -- checking", round_id, error)
            if self._already_submitted(node_id, round_id):
                LOG.info("round %s was accepted after all", round_id)
                return {}
            return self._request(
                "POST", f"/tasks/{node_id}/{round_id}/submit", body=body, retries=0
            )

    # -- server-held shards (contract 0.7.0) ------------------------------

    def get_shard_chunks(
        self,
        shard_id: str,
        *,
        partition: int | None = None,
        n_partitions: int | None = None,
    ) -> dict[str, Any]:
        """The texts this node has to score.

        `partition` and `n_partitions` travel together or not at all -- the
        server answers one without the other with a 422 -- and passing neither
        asks for the whole shard, which is what `experts` and a plain round need.

        An unknown shard is *not* a 404: it comes back 200 with an empty list.
        Emptiness is therefore the caller's to notice, and `shards` does.
        """
        params: dict[str, Any] = {}
        if (partition is None) != (n_partitions is None):
            raise ValueError(
                "partition and n_partitions must be given together "
                f"(got partition={partition}, n_partitions={n_partitions})"
            )
        if partition is not None:
            params["partition"] = partition
            params["n_partitions"] = n_partitions
        return self._request("GET", f"/shards/{shard_id}/chunks", params=params or None)

    def get_shard_labels(self, shard_id: str) -> dict[str, Any]:
        """Every oracle label placed on this shard so far, keyed by chunk_id.

        Shard-wide and campaign-agnostic, and nothing in it says which documents
        are held-out -- see `shards` for why that matters.
        """
        payload = self._request("GET", f"/shards/{shard_id}/labels")
        labels = payload.get("labels") if isinstance(payload, dict) else None
        return labels if isinstance(labels, dict) else {}

    def put_shard_chunks(
        self, shard_id: str, chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Operator endpoint. The node uses it only to seed its own selftest."""
        return self._request(
            "POST", f"/shards/{shard_id}/chunks", body={"chunks": chunks}
        )

    # -- operator endpoints -----------------------------------------------

    def create_round(self, body: dict[str, Any]) -> dict[str, Any]:
        """Operator endpoint, used by selftest to make work for ourselves."""
        return self._request("POST", "/rounds", body=body)

    def create_campaign(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/campaigns", body=body)

    def get_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._request("GET", f"/campaigns/{campaign_id}")

    def advance_campaign(self, campaign_id: str) -> dict[str, Any]:
        """Operator endpoint: consume a finished round and continue or finalize.

        Never called by the agent loop. A node driving its own campaign forward
        would be deciding when its own work counts as done.
        """
        return self._request("POST", f"/campaigns/{campaign_id}/advance")

    def get_round(self, round_id: str) -> dict[str, Any]:
        return self._request("GET", f"/rounds/{round_id}")

    def topk(self, round_id: str, **query: Any) -> dict[str, Any]:
        return self._request("GET", f"/rounds/{round_id}/topk", params=query)

    def telemetry_history(
        self,
        *,
        node_id: str | None = None,
        since: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Stored heartbeat samples, grouped by node (contract 0.9.0).

        Read-only observability: what the node has already reported, useful for
        confirming from the node's side that its telemetry is landing.
        """
        params: dict[str, Any] = {}
        if node_id is not None:
            params["node_id"] = node_id
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        return self._request("GET", "/telemetry/history", params=params or None)

    # -- plumbing ---------------------------------------------------------

    def _already_submitted(self, node_id: str, round_id: str) -> bool:
        """`/tasks` lists rounds not yet submitted, so absence means delivered."""
        try:
            tasks = self.get_tasks(node_id)
        except ApiError:
            return False
        for task in tasks:
            if task.round_id == round_id:
                return task.status == "submitted"
        return True

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        retries: int | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        attempts = self.max_retries if retries is None else retries
        headers = {"Accept": "application/json"}

        data = None
        if body is not None:
            # Explicit, rather than requests' json= : its encoder emits a bare
            # NaN token, which is not valid JSON and fails remotely with an
            # opaque parse error instead of locally with a clear one.
            data = json.dumps(body, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        last: Exception | None = None
        for attempt in range(attempts + 1):
            if attempt:
                time.sleep(_backoff(attempt, last))
                LOG.debug("retrying %s %s (attempt %d)", method, path, attempt + 1)
            try:
                response = self._session.request(
                    method,
                    url,
                    data=data,
                    params=params,
                    headers=headers,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
            except (requests.ConnectionError, requests.Timeout) as error:
                last = TransientError(f"{method} {path}: {error}", url=url)
                continue

            if response.status_code in RETRY_STATUSES:
                last = TransientError(
                    f"{method} {path}: HTTP {response.status_code}",
                    status=response.status_code,
                    url=url,
                )
                continue

            return _interpret(response, method, path, url)

        assert last is not None
        raise last


def _interpret(response: requests.Response, method: str, path: str, url: str) -> Any:
    payload: Any
    try:
        payload = response.json() if response.content else {}
    except ValueError:
        payload = response.text

    if response.ok:
        return payload

    detail = _detail(payload)

    if response.status_code == 404 and "unknown node_id" in detail:
        raise UnknownNodeError(
            f"{method} {path}: {detail}", status=404, detail=detail, body=payload, url=url
        )
    if response.status_code == 409:
        raise RoundClosed(
            f"{method} {path}: {detail}", status=409, detail=detail, body=payload, url=url
        )
    if response.status_code == 422:
        raise ValidationRejected(
            f"{method} {path}: {detail}", status=422, detail=detail, body=payload, url=url
        )
    raise ApiError(
        f"{method} {path}: HTTP {response.status_code} {detail}".strip(),
        status=response.status_code,
        detail=detail,
        body=payload,
        url=url,
    )


def _detail(payload: Any) -> str:
    """Flatten FastAPI's error shapes into one readable line."""
    if isinstance(payload, str):
        return payload[:500]
    if not isinstance(payload, dict):
        return ""
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                where = ".".join(str(p) for p in item.get("loc", []))
                parts.append(f"{where}: {item.get('msg', '')}".strip(": "))
            else:
                parts.append(str(item))
        return "; ".join(parts)[:500]
    return ""


def _backoff(attempt: int, last: Exception | None) -> float:
    if isinstance(last, ApiError) and isinstance(last.body, dict):
        retry_after = last.body.get("retry_after")
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            return min(float(retry_after), BACKOFF_CAP_S)
    # Full jitter: a fleet of nodes reconnecting after a tunnel blip must not
    # arrive in lockstep.
    return random.uniform(0, min(BACKOFF_BASE_S * 2**attempt, BACKOFF_CAP_S))
