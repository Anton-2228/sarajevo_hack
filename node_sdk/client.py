"""Node-side SDK for the Proxy Mesh control plane — stdlib only, so the node can vendor this file.

    cp = ControlPlane("https://cp.example", state_file=".pm_node.json")
    cp.handshake(name="bank-a",
                 hardware={"cpu_cores": 16, "ram_gb": 64, "gpus": [{"model": "A100", "vram_gb": 80}]},
                 model_kinds=["classifier"])
    beat = {"status": "idle"}
    cp.start_heartbeat(lambda: beat)
    for task in cp.tasks():                  # rounds the control plane wants this node in
        cp.ack(task["round_id"])             # "started working"
        beat.update(status="busy", round_id=task["round_id"])
        operation = task["operation"]         # fresh | continue | skip; never infer this locally
        ...  # load/create, train if requested, then score task["dataset_id"]
        cp.submit(task["round_id"], [(chunk_hash, score), ...],
                  {"n_chunks": n, **{m: value_of(m) for m in task["metrics"]},
                   **heldout_curve(heldout_scores, heldout_is_good)})

`submit` is the Egress Gate: only (chunk_hash, score) pairs and numeric metrics leave the node.
`heldout_curve` turns the proxy's ranking of an oracle-labelled held-out split into counts; without it
the control plane can neither calibrate the node's scores nor let it past the Reliability Gate.
Node identity (just node_id — no auth in this hackathon build, see CONTRACT.md §2.1) is saved to
`state_file` so restarts keep the same id.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

Number = Union[int, float]
HELDOUT_QUANTILES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)  # must match control_plane.schemas


def build_payload(node_id: str, round_id: str, scores: Iterable[Tuple[str, Number]],
                  agg_stats: Dict[str, Number]) -> dict:
    score_list = [{"chunk_id": str(cid), "score": float(s)} for cid, s in scores]
    return {"node_id": node_id, "round_id": round_id, "scores": score_list, "agg_stats": dict(agg_stats)}


def heldout_curve(scores: Sequence[Number], good: Sequence[bool]) -> Dict[str, int]:
    """agg_stats keys describing how the proxy ranks its assigned CP-labelled held-out split.

    For each q in HELDOUT_QUANTILES: of the ceil(q * n) best-scored held-out docs, how many there are
    (ho_n_qXX) and how many are good (ho_good_qXX); plus totals ho_n and ho_good. Only counts leave the node.
    """
    n = len(scores)
    if n == 0 or len(good) != n:
        raise ValueError("need one good flag per held-out score, and at least one score")
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    stats = {"ho_n": n, "ho_good": sum(1 for g in good if g)}
    for q in HELDOUT_QUANTILES:
        top = order[:math.ceil(round(q * n, 9))]  # round: 0.1 * 2000 is 200.00000000000003
        key = f"{round(q * 100):02d}"
        stats[f"ho_n_q{key}"] = len(top)
        stats[f"ho_good_q{key}"] = sum(1 for i in top if good[i])
    return stats


class ControlPlaneError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status


class ControlPlane:
    def __init__(self, base_url: str, state_file: Optional[str] = ".pm_node.json", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.state_file = Path(state_file) if state_file else None
        self.node_id: Optional[str] = None
        self.heartbeat_interval_s = 3.0
        if self.state_file and self.state_file.exists():
            self.node_id = json.loads(self.state_file.read_text())["node_id"]

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        headers = {"Content-Type": "application/json", "User-Agent": "proxy-mesh-node/0.11"}
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise ControlPlaneError(exc.code, exc.read().decode()) from exc

    def _require_id(self) -> str:
        if not self.node_id:
            raise RuntimeError("call handshake() first")
        return self.node_id

    # --- handshake / heartbeat ---------------------------------------------------
    def handshake(self, name: str, hardware: Dict[str, Any], model_kinds: Iterable[str],
                  software: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        body = {"name": name, "hardware": hardware, "model_kinds": list(model_kinds),
                "software": software or {}}
        resp = None
        if self.node_id:  # keep identity across restarts
            try:
                resp = self._call("POST", "/nodes/handshake", {**body, "node_id": self.node_id})
            except ControlPlaneError as exc:
                if exc.status != 404:
                    raise
                # saved identity unknown to this control plane (e.g. fresh DB): enroll again
        if resp is None:
            resp = self._call("POST", "/nodes/handshake", body)
        self.node_id = resp["node_id"]
        self.heartbeat_interval_s = float(resp["heartbeat_interval_s"])
        if self.state_file:
            self.state_file.write_text(json.dumps({"node_id": self.node_id}))
        return resp

    def heartbeat(self, status: str = "idle", round_id: Optional[str] = None,
                  load: Optional[Dict[str, Number]] = None, stage: Optional[str] = None) -> Dict[str, Any]:
        """Publish liveness and the latest numeric telemetry snapshot.

        Stable progress keys understood by the built-in dashboard are ``progress_pct``,
        ``docs_processed``, ``docs_total``, ``docs_per_sec``, ``eta_s``, ``train_loss``,
        ``cpu_pct``, ``ram_pct``, ``gpu_util_pct`` and ``gpu_mem_pct``. Custom numeric keys are
        accepted too and exported by ``GET /metrics``. ``stage`` may be ``downloading``,
        ``training``, ``scoring`` or ``uploading``.
        """
        body: Dict[str, Any] = {"status": status, "load": load or {}}
        if round_id:
            body["round_id"] = round_id
        if stage:
            body["stage"] = stage
        return self._call("POST", f"/nodes/{self._require_id()}/heartbeat", body)

    def start_heartbeat(self, get_state: Callable[[], Dict[str, Any]] = dict) -> threading.Event:
        """Daemon thread sending heartbeat(**get_state()) every heartbeat_interval_s. Set the
        returned event to stop it."""
        stop = threading.Event()

        def loop() -> None:
            while True:
                try:
                    response = self.heartbeat(**get_state())
                    next_interval = float(response.get("next_heartbeat_s", self.heartbeat_interval_s))
                    if next_interval > 0:
                        self.heartbeat_interval_s = next_interval
                except Exception as exc:  # keep beating through transient network errors
                    print(f"[proxy-mesh] heartbeat failed: {exc}", file=sys.stderr)
                if stop.wait(self.heartbeat_interval_s):
                    return

        threading.Thread(target=loop, name="pm-heartbeat", daemon=True).start()
        return stop

    # --- tasks ---------------------------------------------------------------------
    def tasks(self) -> List[Dict[str, Any]]:
        """Open rounds the control plane wants this node in (assigned or accepted)."""
        return self._call("GET", f"/tasks/{self._require_id()}")["tasks"]

    def ack(self, round_id: str) -> Dict[str, Any]:
        """Tell the control plane this node has started working on the round."""
        return self._call("POST", f"/tasks/{self._require_id()}/{round_id}/ack")

    # --- Egress Gate -------------------------------------------------------------------
    def submit(self, round_id: str, scores: Iterable[Tuple[str, Number]],
               agg_stats: Dict[str, Number]) -> Dict[str, Any]:
        payload = build_payload(self._require_id(), round_id, scores, agg_stats)
        return self._call("POST", f"/tasks/{self.node_id}/{round_id}/submit", payload)
