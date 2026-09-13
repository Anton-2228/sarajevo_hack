"""Prometheus text exposition for GET /metrics — one read-only pass over the store, no new state.

Point Grafana (or `curl`) straight at it — pull, not push, so no separate metrics pipeline to run.

The one deliberately generic gauge is `proxy_mesh_submission_metric`: whatever numeric key a node
puts in `agg_stats` at submit (eval_spearman, the held-out curve, a custom metric a node invents) shows
up here labelled by its own name, with no server-side allowlist to maintain — a demo node's own numbers
are exactly what ends up on the graph.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from .storage import Store

OFFLINE_AFTER_MISSED_BEATS = 3  # kept in sync with app.py's own constant


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _line(name: str, labels: Dict[str, Any], value: float) -> str:
    tags = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
    return f"{name}{{{tags}}} {value!r}" if tags else f"{name} {value!r}"


class _Family:
    """One metric name: HELP/TYPE once, then any number of labelled samples."""

    def __init__(self, name: str, kind: str, help_text: str):
        self.name, self.kind, self.help_text = name, kind, help_text
        self.samples: List[Tuple[Dict[str, Any], float]] = []

    def add(self, labels: Dict[str, Any], value: float) -> None:
        self.samples.append((labels, value))

    def render(self) -> Iterable[str]:
        if not self.samples:
            return
        yield f"# HELP {self.name} {self.help_text}"
        yield f"# TYPE {self.name} {self.kind}"
        for labels, value in self.samples:
            yield _line(self.name, labels, value)


def render(store: Store, heartbeat_s: float, now: float) -> str:
    families: Dict[str, _Family] = {}

    def fam(name: str, kind: str, help_text: str) -> _Family:
        return families.setdefault(name, _Family(name, kind, help_text))

    # --- nodes: the device map ---------------------------------------------------------
    nodes = store.list_nodes()
    online = fam("proxy_mesh_node_online", "gauge", "1 if the node's last heartbeat is within the online window.")
    last_beat = fam("proxy_mesh_node_last_heartbeat_seconds", "gauge", "Unix time of the last heartbeat.")
    pending = fam("proxy_mesh_node_pending_tasks", "gauge", "Open tasks not yet submitted for this node.")
    gpus = fam("proxy_mesh_node_gpus", "gauge", "GPU count declared at handshake.")
    busy = fam("proxy_mesh_node_busy", "gauge", "1 if the node's latest heartbeat status is busy.")
    heartbeat_age = fam("proxy_mesh_node_heartbeat_age_seconds", "gauge", "Age of the latest heartbeat in seconds.")
    load_fam = fam(
        "proxy_mesh_node_load", "gauge",
        "Numeric telemetry from the node's latest heartbeat, labelled by the client-provided metric name.",
    )
    stage_fam = fam(
        "proxy_mesh_node_stage", "gauge",
        "Current task phase from the node's latest heartbeat (one labelled sample with value 1).",
    )
    for n in nodes:
        labels = {"node_id": n["node_id"], "name": n["name"]}
        is_online = n["last_heartbeat"] is not None and now - n["last_heartbeat"] <= OFFLINE_AFTER_MISSED_BEATS * heartbeat_s
        heartbeat = n["heartbeat"] or {}
        online.add(labels, 1.0 if is_online else 0.0)
        busy.add(labels, 1.0 if heartbeat.get("status") == "busy" else 0.0)
        if heartbeat.get("stage"):
            stage_fam.add({**labels, "stage": heartbeat["stage"]}, 1.0)
        if n["last_heartbeat"] is not None:
            last_beat.add(labels, n["last_heartbeat"])
            heartbeat_age.add(labels, max(0.0, now - n["last_heartbeat"]))
        for key, value in heartbeat.get("load", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                load_fam.add({**labels, "metric": key}, value)
        pending.add(labels, len(store.tasks_for_node(n["node_id"])))
        gpus.add(labels, len(n["specs"].get("hardware", {}).get("gpus", [])))
    fam("proxy_mesh_nodes_total", "gauge", "Registered nodes.").add({}, len(nodes))

    # --- rounds: runs and their participants --------------------------------------------
    rounds = store.list_rounds()
    by_status: Dict[str, int] = {}
    participants_fam = fam("proxy_mesh_round_participants", "gauge", "Participants of a round, by status.")
    metric_fam = fam("proxy_mesh_submission_metric", "gauge",
                     "Numeric agg_stats from a node's latest submission — whatever the node reports.")
    trust_fam = fam("proxy_mesh_submission_trust", "gauge", "1 for the trust status of a node's latest submission.")
    for r in rounds:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        counts: Dict[str, int] = {}
        for p in store.participants_for_round(r["round_id"]):
            counts[p["status"]] = counts.get(p["status"], 0) + 1
        for status, n in counts.items():
            participants_fam.add({"round_id": r["round_id"], "status": status}, n)
        for sub in store.submissions_for_round(r["round_id"]):
            base = {"round_id": r["round_id"], "node_id": sub["node_id"]}
            trust_fam.add({**base, "trust": sub["trust"]}, 1)
            for key, value in sub["agg_stats"].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metric_fam.add({**base, "metric": key}, value)
    for status, n in by_status.items():
        fam("proxy_mesh_rounds_total", "gauge", "Rounds by status.").add({"status": status}, n)

    # --- campaigns: the active learning loop's own progress -------------------------------
    campaigns = store.list_campaigns()
    camp_by_status: Dict[str, int] = {}
    rounds_done = fam(
        "proxy_mesh_campaign_rounds_done", "gauge",
        "Current scheduled round number (legacy name; the current round may still be running).",
    )
    rounds_completed = fam("proxy_mesh_campaign_rounds_completed", "gauge", "Campaign rounds with all submissions.")
    schedule_len = fam("proxy_mesh_campaign_schedule_len", "gauge", "Total rounds this campaign will run.")
    partitions_submitted = fam(
        "proxy_mesh_campaign_partitions_submitted", "gauge", "Participants submitted in the current campaign round."
    )
    campaign_labels = fam("proxy_mesh_campaign_labels", "gauge", "Scheduled oracle labels available to the campaign.")
    campaign_labels_target = fam(
        "proxy_mesh_campaign_labels_target", "gauge", "Target oracle labels at the end of the campaign."
    )
    selected = fam("proxy_mesh_campaign_selected", "gauge", "Chunks in the final selection (0 until done).")
    for c in campaigns:
        camp_by_status[c["status"]] = camp_by_status.get(c["status"], 0) + 1
        labels = {"campaign_id": c["campaign_id"], "shard_id": c["shard_id"]}
        schedule = c["spec"]["schedule"]
        n_partitions = c["spec"].get("n_partitions", len(c["spec"]["node_ids"]))
        current_round = store.current_campaign_round(c["campaign_id"])
        submitted = (sum(p["status"] == "submitted" for p in store.participants_for_round(current_round))
                     if current_round and c["status"] == "running" else n_partitions)
        current_complete = submitted == n_partitions
        completed = (c["rounds_done"] if c["status"] == "done" or current_complete
                     else max(c["rounds_done"] - 1, 0))
        n_labels = (schedule[min(c["rounds_done"], len(schedule)) - 1] * n_partitions
                    if c["rounds_done"] else 0)
        rounds_done.add(labels, c["rounds_done"])
        rounds_completed.add(labels, completed)
        schedule_len.add(labels, len(schedule))
        partitions_submitted.add(labels, submitted)
        campaign_labels.add(labels, n_labels)
        campaign_labels_target.add(labels, schedule[-1] * n_partitions)
        if c["result"]:
            selected.add(labels, len(c["result"]["selected"]))
    for status, n in camp_by_status.items():
        fam("proxy_mesh_campaigns_total", "gauge", "Campaigns by status.").add({"status": status}, n)

    # --- shards: the pool the server holds -------------------------------------------------
    chunks_fam = fam("proxy_mesh_shard_chunks_total", "gauge", "Chunks loaded into a shard.")
    labels_fam = fam("proxy_mesh_shard_labels_total", "gauge", "Oracle labels produced so far for a shard.")
    for shard_id in store.list_shard_ids():
        chunks_fam.add({"shard_id": shard_id}, len(store.chunk_ids_for_shard(shard_id)))
        labels_fam.add({"shard_id": shard_id}, len(store.labels_for_shard(shard_id)))

    lines = [line for f in families.values() for line in f.render()]
    return "\n".join(lines) + "\n"


def snapshot(store: Store, heartbeat_s: float, now: float) -> Dict[str, Any]:
    """The same read, shaped as nested JSON for GET /metrics.json — what dashboard.html polls.

    Independent of render() rather than sharing its flat gauge list: a UI wants submissions nested
    under their round and a node's declared model_kinds alongside its online flag, which don't fit the
    one-name-one-number Prometheus shape. Small enough duplication of the store reads to keep rather
    than force one shape to serve both."""
    nodes = []
    for n in store.list_nodes():
        is_online = (n["last_heartbeat"] is not None
                    and now - n["last_heartbeat"] <= OFFLINE_AFTER_MISSED_BEATS * heartbeat_s)
        heartbeat = n["heartbeat"] or {}
        nodes.append({
            "node_id": n["node_id"], "name": n["name"], "online": is_online,
            "last_heartbeat": n["last_heartbeat"], "pending_tasks": len(store.tasks_for_node(n["node_id"])),
            "model_kinds": n["specs"].get("model_kinds", []),
            "gpus": len(n["specs"].get("hardware", {}).get("gpus", [])),
            "status": heartbeat.get("status", "idle"), "stage": heartbeat.get("stage"),
            "round_id": heartbeat.get("round_id"),
            "load": {k: v for k, v in heartbeat.get("load", {}).items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)},
            "heartbeat_age_s": (max(0.0, now - n["last_heartbeat"])
                                if n["last_heartbeat"] is not None else None),
        })

    rounds = []
    for r in store.list_rounds():
        counts: Dict[str, int] = {}
        for p in store.participants_for_round(r["round_id"]):
            counts[p["status"]] = counts.get(p["status"], 0) + 1
        submissions = [{"node_id": s["node_id"], "trust": s["trust"], "n_scores": s["n_scores"],
                       "agg_stats": {k: v for k, v in s["agg_stats"].items()
                                    if isinstance(v, (int, float)) and not isinstance(v, bool)}}
                      for s in store.submissions_for_round(r["round_id"])]
        rounds.append({"round_id": r["round_id"], "status": r["status"],
                       "model": (r["spec"] or {}).get("model"),
                       "operation": (r["spec"] or {}).get("operation"), "participants": counts,
                       "n_participants": sum(counts.values()), "submissions": submissions})

    campaigns = []
    for c in store.list_campaigns():
        schedule = c["spec"]["schedule"]
        n_partitions = c["spec"].get("n_partitions", len(c["spec"]["node_ids"]))
        current_round = store.current_campaign_round(c["campaign_id"])
        submitted = (sum(p["status"] == "submitted" for p in store.participants_for_round(current_round))
                     if current_round and c["status"] == "running" else n_partitions)
        current_complete = submitted == n_partitions
        completed = (c["rounds_done"] if c["status"] == "done" or current_complete
                     else max(c["rounds_done"] - 1, 0))
        campaigns.append({
            "campaign_id": c["campaign_id"], "shard_id": c["shard_id"], "status": c["status"],
            "mode": c["spec"].get("mode", "sharded"), "n_partitions": n_partitions,
            "train_mode": c["spec"].get("train_mode", "fresh"),
            "partitions_submitted": submitted,
            "strategy": c["spec"]["strategy"], "rounds_done": c["rounds_done"],
            "current_step": c["rounds_done"], "completed_rounds": completed, "schedule_len": len(schedule),
            "n_labels": (schedule[min(c["rounds_done"], len(schedule)) - 1] * n_partitions
                         if c["rounds_done"] else 0),
            "n_labels_target": schedule[-1] * n_partitions,
            "n_selected": len(c["result"]["selected"]) if c["result"] else None,
        })

    shards = [{"shard_id": s, "n_chunks": len(store.chunk_ids_for_shard(s)), "n_labels": len(store.labels_for_shard(s))}
             for s in store.list_shard_ids()]

    return {"generated_at": now, "nodes": nodes, "rounds": rounds, "campaigns": campaigns, "shards": shards}
