"""The agent's main loop, and the two commands that drive the server directly.

Structure: one background thread heartbeats on the server's schedule, and the
main thread does poll -> ack -> run -> submit strictly one round at a time. No
asyncio, because there is exactly one concurrent thing happening and a thread
says so more plainly.

The ordering that matters is in `_process`: the finished payload reaches disk
before it reaches the network, and is deleted only once delivery is confirmed.
Everything in `_resume` follows from that.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
import uuid
from typing import Any

from node.agent import __version__, resources
from node.agent.api import (
    ApiError,
    ControlPlaneClient,
    RoundClosed,
    TransientError,
    UnknownNodeError,
    ValidationRejected,
)
from node.agent.config import AgentConfig
from node.agent.models import (
    STAGE_UPLOADING,
    AgentView,
    as_label,
    HandshakeRequest,
    Hardware,
    SubmitPayload,
    TaskProgress,
    TaskView,
    resolve_mode,
)
from node.agent.reporting import ConsoleReporter, Reporter
from node.agent.runner import TaskFailed, TaskRunner
from node.agent.shards import ShardCorpusSource
from node.agent.telemetry import Telemetry
from node.agent.state import (
    MAX_TASK_ATTEMPTS,
    PHASE_ACKED,
    PHASE_FAILED,
    PHASE_READY,
    PHASE_SUBMITTED,
    AgentState,
    Identity,
    TaskRecord,
)

LOG = logging.getLogger("node.agent.loop")

REENROL_COOLDOWN_S = 30.0
HEARTBEAT_FAILURES_BEFORE_DEGRADED = 3

# Applied to intervals the *server* hands us, so that a misconfigured or buggy
# control plane cannot talk a fleet of nodes into hammering it. An interval the
# operator passed explicitly is their own call and is taken as given.
MIN_SERVER_HEARTBEAT_S = 1.0
# Only guards against a spin; not a policy.
MIN_HEARTBEAT_S = 0.01


class Heartbeater(threading.Thread):
    """Tells the server we are alive, and wakes the main loop when work lands.

    It reads a snapshot the work loop maintains and posts it on the server's own
    interval (METRICS_GUIDE). It never computes telemetry itself: doing that here
    would sample the machine from the wrong thread and at the wrong moment.
    """

    def __init__(
        self,
        client: ControlPlaneClient,
        node_id: str,
        telemetry: Telemetry,
        *,
        interval_s: float,
        wake: threading.Event,
        reenrol: threading.Event,
    ) -> None:
        super().__init__(name="heartbeat", daemon=True)
        self._client = client
        self._node_id = node_id
        self._telemetry = telemetry
        self._interval = max(interval_s, MIN_HEARTBEAT_S)
        self._wake = wake
        self._reenrol = reenrol
        self._stop = threading.Event()
        # `beat_now` is called from the main thread while this thread may be
        # mid-request, and requests.Session is not documented thread-safe.
        self._sending = threading.Lock()
        self.consecutive_failures = 0
        self.last_ok = False

    def stop(self) -> None:
        self._stop.set()

    def beat_now(self) -> None:
        """Send one beat immediately, outside the schedule.

        Worth a request when a task ends: the guide allows it, and it keeps the
        dashboard from showing a finished round as still running for up to a
        whole interval. Never raises -- an out-of-band nicety must not fail a
        round that already succeeded.
        """
        try:
            with self._sending:
                self._client.heartbeat(self._node_id, self._telemetry.snapshot())
        except ApiError as error:
            LOG.debug("out-of-band heartbeat failed: %s", error)

    def run(self) -> None:
        # Event.wait, not time.sleep: it is interruptible, and it behaves the
        # same on Windows, where sleep cannot be broken by a signal.
        while not self._stop.wait(self._interval):
            try:
                with self._sending:
                    response = self._client.heartbeat(
                        self._node_id, self._telemetry.snapshot()
                    )
            except UnknownNodeError:
                # The main thread owns identity.json; two threads racing to
                # rewrite it would be worse than a slightly delayed recovery.
                LOG.warning("heartbeat: server no longer knows this node")
                self._reenrol.set()
                self._wake.set()
                continue
            except ApiError as error:
                self.consecutive_failures += 1
                self.last_ok = False
                LOG.warning("heartbeat failed (%d in a row): %s", self.consecutive_failures, error)
                continue

            self.consecutive_failures = 0
            self.last_ok = True
            self._interval = max(response.next_heartbeat_s, MIN_SERVER_HEARTBEAT_S)
            if response.pending_tasks > 0:
                # Do not sit out the poll interval when the server has said
                # outright that there is work.
                self._wake.set()


class AgentLoop:
    def __init__(
        self,
        config: AgentConfig,
        reporter: Reporter | None = None,
        client: ControlPlaneClient | None = None,
    ) -> None:
        self.config = config
        self.reporter = reporter or ConsoleReporter()
        self.stop_event = threading.Event()

        self._client = client or ControlPlaneClient(
            config.server_url,
            connect_timeout=config.connect_timeout_s,
            read_timeout=config.read_timeout_s,
            max_retries=config.max_retries,
        )
        self._state = AgentState(config.state_dir)
        self._wake = threading.Event()
        self._reenrol = threading.Event()
        self._heart: Heartbeater | None = None
        self._identity: Identity | None = None
        self._last_enrol_attempt = 0.0

        hw = resources.detect_hardware(config.state_dir.parent)
        self._hardware = hw
        self._budget, budget_warnings = resources.resolve_budget(
            config.cores, config.ram_gb, hw
        )
        for warning in budget_warnings:
            self.reporter.note(logging.WARNING, warning)

        # Contract 0.7.0: the corpus is server-held, so the runner reads it off
        # the control plane rather than off this machine. Nothing is provisioned
        # locally any more, and nothing is substituted -- scoring chunk ids the
        # server did not assign is worse than failing the round, because in a
        # sharded campaign it corrupts the merge for every other node.
        self._corpus_source = ShardCorpusSource(self._client)

        # The snapshot the heartbeat thread posts. The runner writes stages and
        # progress into it; nothing in the work path touches the network.
        self._telemetry = Telemetry(self._budget)

        self._runner = TaskRunner(
            state=self._state,
            corpus_source=self._corpus_source,
            budget=self._budget,
            reporter=self.reporter,
            score_scale=config.score_scale,
            unknown_metric=config.unknown_metric,
            enforce_limits=config.enforce_limits,
            verbose=2 if config.log_level == "debug" else 0,
            progress=self._telemetry,
        )

        self.view = AgentView(
            server_url=config.server_url,
            budget_cores=self._budget.cores,
            budget_ram_gb=round(self._budget.ram_gb, 2),
        )

    # -- lifecycle --------------------------------------------------------

    def request_stop(self) -> None:
        """What the signal handler and a future GUI Stop button both call."""
        self.stop_event.set()
        self._wake.set()

    def run(self) -> int:
        self._banner()

        try:
            self._client.health()
        except ApiError as error:
            # Fail loudly on a bad URL rather than through a confusing
            # handshake error three calls later.
            self.reporter.note(logging.ERROR, f"control plane unreachable: {error}")
            return 1

        if self.config.reset_identity:
            self._state.clear_identity()
        if not self._enrol():
            return 1

        resources.prime_cpu_percent()
        self._state.prune()
        self._start_heartbeat()

        try:
            self._resume()
            return self._poll_forever()
        finally:
            if self._heart:
                self._heart.stop()
            self.view.status = "stopping"
            self.reporter.state(self.view)

    def _banner(self) -> None:
        self.reporter.note(
            logging.INFO,
            f"node-agent {__version__} starting",
            server=self.config.server_url,
            platform=self._hardware.platform,
            cores=self._budget.cores,
            ram_gb=round(self._budget.ram_gb, 2),
            state_dir=str(self.config.state_dir),
            enforce_limits=self.config.enforce_limits,
        )
        self.reporter.note(
            logging.INFO,
            "workloads are server-held; corpora are read from /shards per round",
        )

    # -- enrolment --------------------------------------------------------

    def _enrol(self, *, force_new: bool = False) -> bool:
        stored = None if force_new else self._state.load_identity(self.config.server_url)

        request = HandshakeRequest(
            name=self.config.name,
            node_id=stored.node_id if stored else None,
            hardware=Hardware(
                cpu_cores=self._budget.cores,
                ram_gb=round(self._budget.ram_gb, 2),
                gpus=[],
                disk_free_gb=self._hardware.disk_free_gb,
            ),
            # Values must match the contract's LABEL format; ours already do,
            # but platform strings are not ours to guarantee.
            software={
                "agent_version": as_label(__version__),
                "python": as_label(self._hardware.python),
                "platform": as_label(self._hardware.platform),
            },
            # Only what this node can actually bake. There is no torch here, so
            # claiming "lora" would earn us rounds we would have to fail.
            model_kinds=list(self.config.model_kinds),
        )

        self._last_enrol_attempt = time.time()
        try:
            response = self._client.handshake(request)
        except ApiError as error:
            if stored is not None:
                # Re-registering with our old id was refused; take a new one.
                LOG.warning("re-registration refused (%s); enrolling fresh", error)
                return self._enrol(force_new=True)
            self.reporter.note(logging.ERROR, f"handshake failed: {error}")
            return False

        identity = Identity(
            node_id=response.node_id,
            server_url=self.config.server_url,
            name=self.config.name,
            heartbeat_interval_s=response.heartbeat_interval_s,
        )
        self._state.save_identity(identity)
        self._identity = identity

        self.view.node_id = identity.node_id
        self.view.status = "idle"
        self.reporter.note(
            logging.INFO,
            "enrolled",
            node_id=identity.node_id,
            heartbeat_s=identity.heartbeat_interval_s,
        )
        return True

    def _reenrol_if_needed(self) -> None:
        if not self._reenrol.is_set():
            return
        self._reenrol.clear()
        if time.time() - self._last_enrol_attempt < REENROL_COOLDOWN_S:
            return
        LOG.warning("re-enrolling with the control plane")
        if self._enrol() and self._heart:
            self._heart.stop()
            self._start_heartbeat()

    def _start_heartbeat(self) -> None:
        assert self._identity is not None
        interval = self.config.heartbeat_interval_s or self._identity.heartbeat_interval_s
        # A separate client: requests.Session is not documented thread-safe and
        # this runs alongside a submit that can take minutes.
        beat_client = self._client.clone()
        self._heart = Heartbeater(
            beat_client,
            self._identity.node_id,
            self._telemetry,
            interval_s=interval,
            wake=self._wake,
            reenrol=self._reenrol,
        )
        self._heart.start()

    # -- the loop ---------------------------------------------------------

    def _poll_forever(self) -> int:
        assert self._identity is not None
        completed = 0

        while not self.stop_event.is_set():
            self._reenrol_if_needed()

            try:
                tasks = self._client.get_tasks(self._identity.node_id)
            except UnknownNodeError:
                self._reenrol.set()
                continue
            except ApiError as error:
                self.reporter.note(logging.WARNING, f"poll failed: {error}")
                self._sleep_until_woken()
                continue

            # Oldest first, and a round already submitted is not work.
            todo = sorted(
                (t for t in tasks if t.status != "submitted"), key=lambda t: t.assigned_at
            )
            if todo:
                self.reporter.note(logging.INFO, "poll returned work", tasks=len(todo))

            for task in todo:
                if self.stop_event.is_set():
                    break
                if self._should_skip(task):
                    continue
                if self._process(task):
                    completed += 1
                    self.view.completed = completed
                else:
                    self.view.failed += 1
                self.reporter.state(self.view)

                if self.config.max_tasks and completed >= self.config.max_tasks:
                    return 0

            # --once means "one pass, then exit", and that has to hold even when
            # the pass handled nothing -- every round skipped, or none offered.
            # Otherwise a node whose only round has exhausted its attempts polls
            # that same round forever.
            if self.config.once:
                if not completed:
                    self.reporter.note(
                        logging.INFO, "nothing to do this pass; --once so exiting"
                    )
                return 0

            self._sleep_until_woken()

        return 0

    def _sleep_until_woken(self) -> None:
        self._wake.wait(self.config.poll_interval_s)
        self._wake.clear()
        if self._heart and self._heart.consecutive_failures >= HEARTBEAT_FAILURES_BEFORE_DEGRADED:
            self.view.status = "degraded"
        self.view.last_heartbeat_ok = bool(self._heart and self._heart.last_ok)

    def _should_skip(self, task: TaskView) -> bool:
        record = self._state.get_record(task.round_id)
        if record and record.phase == PHASE_FAILED and record.attempts >= MAX_TASK_ATTEMPTS:
            # Otherwise a permanently broken round becomes an infinite
            # train-fail-retry loop that eats the machine.
            LOG.debug("skipping round %s after %d attempts", task.round_id, record.attempts)
            return True
        return False

    def _process(self, task: TaskView) -> bool:
        assert self._identity is not None
        node_id = self._identity.node_id
        record = self._state.get_record(task.round_id) or TaskRecord(
            round_id=task.round_id, phase=PHASE_ACKED
        )
        started = time.time()

        self._set_busy(task.round_id)
        self.view.current = TaskProgress(task.round_id, "acked", started)
        try:
            if self.config.dry_run:
                self.reporter.note(logging.INFO, "dry run: not acking", round_id=task.round_id)
            else:
                self._client.ack(node_id, task.round_id)
            record.phase = PHASE_ACKED
            record.dataset_id = task.dataset_id
            self._state.record(record)

            for problem in task.routing_problems:
                # Logged before the runner refuses the round, so the reason is
                # visible even if the failure is read from the journal later.
                self.reporter.note(
                    logging.ERROR,
                    "task routing is unusable",
                    round_id=task.round_id,
                    problem=problem,
                )

            mode = resolve_mode(task, self.config.mode)
            record.train_policy = mode.policy
            self._state.record(record)
            self.reporter.task(
                TaskProgress(
                    task.round_id,
                    "running",
                    started,
                    {
                        "mode": mode.source,
                        "operation": task.operation.describe(),
                        "shard_id": task.shard_id,
                        "routing": task.routing.describe(),
                    },
                )
            )
            outcome = self._runner.run(task, mode)
            for warning in outcome.warnings:
                self.reporter.note(logging.WARNING, warning, round_id=task.round_id)

            record.model_key = outcome.model_key
            record.checkpoint_id = outcome.checkpoint_id
            payload = SubmitPayload(
                node_id=node_id,
                round_id=task.round_id,
                scores=outcome.scores,
                agg_stats=outcome.agg_stats,
            )
            submit_started = time.time()
            self._deliver(task.round_id, payload, record)
            submit_s = time.time() - submit_started

            record.phase = PHASE_SUBMITTED
            record.last_error = None
            self._state.record(record)
            self._state.drop_payload(task.round_id)
            self.reporter.task(
                TaskProgress(
                    task.round_id,
                    "done",
                    started,
                    {
                        "n_scores": len(outcome.scores),
                        "seconds": round(time.time() - started, 1),
                        # The network half of the round versus this machine's,
                        # so an operator can tell a slow link from a slow box.
                        "download_s": outcome.timings.get("download_s"),
                        "compute_s": outcome.timings.get("compute_s"),
                        "submit_s": round(submit_s, 3),
                    },
                )
            )
            return True

        except TaskFailed as error:
            self._fail(record, error, permanent=error.permanent)
            return False
        except RoundClosed as error:
            # The operator ended the round. There is no reopening it, so the
            # work is gone and a retry would only waste the machine again.
            self._fail(record, error, permanent=True)
            return False
        except ApiError as error:
            self._fail(record, error, permanent=isinstance(error, ValidationRejected))
            return False
        except Exception as error:  # noqa: BLE001 - one bad round must not kill the node
            if self.stop_event.is_set():
                # On Windows a console Ctrl+C reaches the training child too,
                # which dies mid-run. That is a shutdown, not a failure.
                self.reporter.note(
                    logging.INFO, "round aborted by shutdown", round_id=task.round_id
                )
                return False
            LOG.exception("round %s failed unexpectedly", task.round_id)
            self._fail(record, error, permanent=False)
            return False
        finally:
            self._set_busy(None)
            self.view.current = None

    def _deliver(self, round_id: str, payload: SubmitPayload, record: TaskRecord) -> None:
        assert self._identity is not None
        identity = self._identity
        body = payload.body()

        # The server rejects a mismatch, and finding out after the training is
        # a needless way to lose a round.
        assert body["node_id"] == identity.node_id
        assert body["round_id"] == round_id

        if self.config.dry_run:
            self.reporter.note(
                logging.INFO,
                "dry run: would submit",
                round_id=round_id,
                n_scores=len(body["scores"]),
                payload={**body, "scores": body["scores"][:3]},
            )
            return

        # On disk before the wire. A crash in this window then costs a POST,
        # not a retrain.
        self._state.stash_payload(round_id, body)
        record.phase = PHASE_READY
        self._state.record(record)

        self._telemetry.stage(STAGE_UPLOADING)
        self._telemetry.progress(len(body["scores"]), len(body["scores"]))

        try:
            self._client.submit(identity.node_id, round_id, body)
        except ValidationRejected as error:
            retried = self._retry_missing_metrics(round_id, payload, error)
            if retried is None:
                raise
            body = retried

        self.reporter.note(
            logging.INFO,
            "submitted",
            round_id=round_id,
            n_scores=len(body["scores"]),
            bytes=len(str(body)),
        )

    def _retry_missing_metrics(
        self, round_id: str, payload: SubmitPayload, error: ValidationRejected
    ) -> dict[str, Any] | None:
        """The round demanded metrics this node could not compute. Send zeros once.

        The default policy leaves an uncomputable metric out, so the server gets
        to say whether it really needs it. Only when it says so do we send a
        number we do not believe -- and say loudly that we did. One HTTP round
        trip against a lost training run.
        """
        assert self._identity is not None
        missing = [key for key in error.missing_metrics if key not in payload.agg_stats]
        if not missing:
            return None

        self.reporter.note(
            logging.WARNING,
            "round requires metrics this node cannot compute; submitting zeros for them",
            round_id=round_id,
            fields=",".join(missing),
        )
        patched = SubmitPayload(
            node_id=payload.node_id,
            round_id=payload.round_id,
            scores=payload.scores,
            agg_stats={**payload.agg_stats, **{k: 0.0 for k in missing}},
        )
        body = patched.body()
        self._state.stash_payload(round_id, body)
        self._client.submit(self._identity.node_id, round_id, body)
        return body

    def _fail(self, record: TaskRecord, error: Exception, *, permanent: bool) -> None:
        record.phase = PHASE_FAILED
        record.attempts = MAX_TASK_ATTEMPTS if permanent else record.attempts + 1
        record.last_error = f"{type(error).__name__}: {error}"
        self._state.record(record)
        self.view.last_error = record.last_error
        self.reporter.note(
            logging.ERROR,
            "round failed" + (" permanently" if permanent else ""),
            round_id=record.round_id,
            attempts=record.attempts,
            error=record.last_error,
        )

    def _set_busy(self, round_id: str | None) -> None:
        self.view.status = "busy" if round_id else "idle"
        if round_id:
            self._telemetry.begin(round_id)
        else:
            # Finishing a task drops the round, the stage and every gauge from
            # it: a stale progress_pct on an idle node is a lie the dashboard
            # would draw (METRICS_GUIDE).
            self._telemetry.idle()
            if self._heart:
                # One immediate beat so the dashboard does not keep showing the
                # finished round as still running until the next tick.
                self._heart.beat_now()

    # -- resume -----------------------------------------------------------

    def _resume(self) -> None:
        """Finish what a previous run started."""
        assert self._identity is not None
        for round_id, record in self._state.journal().items():
            if record.phase == PHASE_READY:
                stashed = self._state.load_payload(round_id)
                if stashed is None:
                    continue
                # Complete and already paid for: send it, do not retrain.
                self.reporter.note(
                    logging.INFO, "resending a payload left by a previous run", round_id=round_id
                )
                try:
                    self._client.submit(self._identity.node_id, round_id, stashed)
                except ApiError as error:
                    self.reporter.note(
                        logging.WARNING, f"resend failed: {error}", round_id=round_id
                    )
                    continue
                record.phase = PHASE_SUBMITTED
                self._state.record(record)
                self._state.drop_payload(round_id)
            elif record.phase == PHASE_FAILED and record.attempts >= MAX_TASK_ATTEMPTS:
                self.reporter.note(
                    logging.WARNING,
                    "skipping a round that failed too many times",
                    round_id=round_id,
                    error=record.last_error,
                )


# -- entry points ----------------------------------------------------------


def _install_signal_handlers(loop: AgentLoop) -> None:
    state = {"count": 0}

    def handler(signum, frame):  # noqa: ARG001
        # Exactly one thing, and nothing that takes a lock: a handler that
        # touches logging while the interrupted thread holds its lock deadlocks.
        state["count"] += 1
        if state["count"] > 1:
            os._exit(130)
        loop.request_stop()

    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)
    if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handler)


def run_agent(config: AgentConfig) -> int:
    loop = AgentLoop(config)
    _install_signal_handlers(loop)
    try:
        return loop.run()
    except KeyboardInterrupt:
        return 130


SELFTEST_SCHEDULE = (20, 40)


def run_selftest(config: AgentConfig, round_id: str | None = None) -> int:
    """One complete round against the live server, created by us, for us.

    The operator endpoints are open, so a node can manufacture its own work and
    prove the whole path end to end. Since contract 0.7.0 that means a *campaign*
    rather than a bare round: oracle labels are placed by the campaign schedule,
    and `GET /shards/{id}/labels` on a shard no campaign has touched is empty --
    a plain `POST /rounds` would hand us a corpus with nothing to train on.

    So: upload a synthetic shard, start a one-node campaign over it, let the
    normal loop do the work, then read the result back. Never closes the round
    and never advances the campaign past what it needs: both are the operator's
    call, not ours.
    """
    from dataclasses import replace

    from node.agent.models import as_id

    tag = uuid.uuid4().hex[:8]
    campaign_id = as_id(round_id or f"selftest-{tag}")
    shard_id = f"{campaign_id}-shard"

    loop = AgentLoop(replace(config, once=True))
    _install_signal_handlers(loop)

    reporter = loop.reporter
    try:
        loop._client.health()
    except ApiError as error:
        reporter.note(logging.ERROR, f"control plane unreachable: {error}")
        return 1

    if config.reset_identity:
        loop._state.clear_identity()
    if not loop._enrol():
        return 1
    assert loop._identity is not None

    # A shard to work on. Built locally, uploaded, and from then on read back
    # off the server like any other workload -- including its chunk ids, so the
    # selftest exercises the same path a real round takes.
    chunks = _synthetic_shard(loop.config.state_dir)
    if len(chunks) < SELFTEST_SCHEDULE[-1]:
        reporter.note(
            logging.ERROR,
            f"synthetic shard has only {len(chunks)} chunks; a campaign partition "
            f"needs at least {SELFTEST_SCHEDULE[-1]}",
        )
        return 1

    reporter.note(logging.INFO, "uploading a shard", shard_id=shard_id, chunks=len(chunks))
    try:
        for start in range(0, len(chunks), 5_000):
            loop._client.put_shard_chunks(shard_id, chunks[start : start + 5_000])
    except ApiError as error:
        reporter.note(logging.ERROR, f"could not upload the shard: {error}")
        return 1

    reporter.note(
        logging.INFO,
        "creating a campaign for ourselves",
        campaign_id=campaign_id,
        shard_id=shard_id,
        mode="sharded",
    )
    try:
        loop._client.create_campaign(
            {
                "campaign_id": campaign_id,
                "shard_id": shard_id,
                "mode": "sharded",
                "model": {"kind": "classifier", "id": f"selftest-{tag}"},
                "metrics": ["eval_spearman", "n_dedup_dropped"],
                "node_ids": [loop._identity.node_id],
                "schedule": list(SELFTEST_SCHEDULE),
                # The campaign's policy for every round it generates. `continue`
                # makes round 1 fresh and chains the rest to the preceding
                # checkpoint, which is the path worth exercising -- the node
                # still obeys each task's own `operation`, not this field.
                "train_mode": "continue",
                "strategy": "cutoff",
                "k_frac": 0.1,
                "seed": 0,
            }
        )
    except ApiError as error:
        reporter.note(logging.ERROR, f"could not create the test campaign: {error}")
        return 1

    resources.prime_cpu_percent()
    loop._start_heartbeat()
    try:
        exit_code = loop._poll_forever()
    finally:
        if loop._heart:
            loop._heart.stop()

    try:
        reporter.note(
            logging.INFO, "campaign state", **_flat(loop._client.get_campaign(campaign_id))
        )
    except ApiError as error:
        reporter.note(logging.WARNING, f"could not read the campaign back: {error}")

    return exit_code


def _synthetic_shard(state_dir: Any) -> list[dict[str, str]]:
    """The golden set, shaped as shard chunks the server will accept.

    Chunk ids are content hashes, so the ids we upload are the ids the server
    hands back -- the selftest then reads its own corpus the same way a real
    round does, rather than through a shortcut that proves less.
    """
    import tempfile
    from pathlib import Path

    from node.agent.datasets import load_dataset, synthesize_golden

    with tempfile.TemporaryDirectory(dir=state_dir if state_dir.is_dir() else None) as tmp:
        path = synthesize_golden(Path(tmp) / "selftest.jsonl")
        return load_dataset(path, "selftest").as_shard_chunks()


def _flat(payload: Any) -> dict[str, Any]:
    """Flatten a server response into loggable scalars."""
    if not isinstance(payload, dict):
        return {"value": payload}
    out: dict[str, Any] = {}
    for key, value in payload.items():
        out[key] = len(value) if isinstance(value, (list, dict)) else value
    return out
