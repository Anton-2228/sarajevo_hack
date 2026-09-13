"""SQLite catalog for the control plane: nodes, rounds, participants, submissions, chunk scores."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id        TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    specs          TEXT NOT NULL,              -- hardware, software, model_kinds
    registered_at  REAL NOT NULL,
    updated_at     REAL NOT NULL,
    last_heartbeat REAL,
    heartbeat      TEXT                        -- last heartbeat body
);
CREATE TABLE IF NOT EXISTS heartbeat_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id     TEXT NOT NULL REFERENCES nodes(node_id),
    received_at REAL NOT NULL,
    status      TEXT NOT NULL,
    stage       TEXT,
    round_id    TEXT,
    load        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_samples_node_time
    ON heartbeat_samples(node_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_heartbeat_samples_time ON heartbeat_samples(received_at);
CREATE TABLE IF NOT EXISTS rounds (
    round_id   TEXT PRIMARY KEY,
    budget_k   INTEGER NOT NULL,
    note       TEXT,
    status     TEXT NOT NULL DEFAULT 'open',   -- open | closed
    created_at REAL NOT NULL,
    spec       TEXT                            -- model, metrics, params
);
CREATE TABLE IF NOT EXISTS participants (
    round_id     TEXT NOT NULL REFERENCES rounds(round_id),
    node_id      TEXT NOT NULL REFERENCES nodes(node_id),
    dataset_id   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'assigned',  -- assigned | accepted | submitted
    assigned_at  REAL NOT NULL,
    accepted_at  REAL,
    submitted_at REAL,
    params       TEXT,                              -- per-node overrides of round params
    PRIMARY KEY (round_id, node_id)
);
CREATE TABLE IF NOT EXISTS submissions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id       TEXT NOT NULL REFERENCES rounds(round_id),
    node_id        TEXT NOT NULL,
    revision       INTEGER NOT NULL,
    received_at    REAL NOT NULL,
    n_scores       INTEGER NOT NULL,
    eval_spearman  REAL,
    trust          TEXT NOT NULL,              -- pending (gated at top-k) | unknown (no held-out curve)
    agg_stats      TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    UNIQUE (round_id, node_id)
);
CREATE TABLE IF NOT EXISTS scores (
    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    chunk_id      TEXT NOT NULL,
    score         REAL NOT NULL,
    PRIMARY KEY (submission_id, chunk_id)
) WITHOUT ROWID;
-- Server-master (campaigns, see campaigns.py): the server holds the pool and the oracle, nodes only
-- train/score. A chunk's text lives here, not on any node.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  TEXT PRIMARY KEY,
    shard_id  TEXT NOT NULL,
    text      TEXT NOT NULL,
    added_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_shard ON chunks(shard_id);
CREATE TABLE IF NOT EXISTS labels (
    chunk_id   TEXT PRIMARY KEY REFERENCES chunks(chunk_id),
    label      INTEGER NOT NULL,
    source     TEXT NOT NULL,          -- which oracle produced it
    labeled_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    shard_id    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',  -- running | done
    spec        TEXT NOT NULL,                    -- mode, partitions, model, metrics, schedule, strategy, k_frac
    rounds_done INTEGER NOT NULL DEFAULT 0,
    result      TEXT,                             -- set once status='done': selected chunk_ids + score
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS campaign_rounds (
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    round_id    TEXT NOT NULL REFERENCES rounds(round_id),
    step        INTEGER NOT NULL,               -- 1-based: which schedule entry this round trains on
    PRIMARY KEY (campaign_id, round_id)
);
"""

TASK_SELECT = """
SELECT p.round_id, p.node_id, p.dataset_id, p.status, p.assigned_at, p.accepted_at,
       p.submitted_at, p.params AS participant_params, r.budget_k, r.status AS round_status, r.spec
FROM participants p JOIN rounds r ON r.round_id = p.round_id
"""


def _loads(value: Optional[str]) -> Any:
    return json.loads(value) if value else None


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        # DBs created before rounds carried a spec.
        if "spec" not in {r["name"] for r in self._conn.execute("PRAGMA table_info(rounds)")}:
            self._conn.execute("ALTER TABLE rounds ADD COLUMN spec TEXT")
        # ... and before participants carried per-node params.
        if "params" not in {r["name"] for r in self._conn.execute("PRAGMA table_info(participants)")}:
            self._conn.execute("ALTER TABLE participants ADD COLUMN params TEXT")
        # Nodes are compute-only. Remove dataset declarations left by pre-0.7 databases.
        with self._conn:
            for row in self._conn.execute("SELECT node_id, specs FROM nodes").fetchall():
                specs = json.loads(row["specs"])
                if "datasets" in specs:
                    specs.pop("datasets")
                    self._conn.execute(
                        "UPDATE nodes SET specs=? WHERE node_id=?",
                        (json.dumps(specs, sort_keys=True), row["node_id"]),
                    )
            # Task operation became explicit in 0.9. Existing rounds used the same implicit rule:
            # train a fresh model from the recipe, save it under the round id, then score.
            for row in self._conn.execute("SELECT round_id, spec FROM rounds").fetchall():
                spec = json.loads(row["spec"]) if row["spec"] else {}
                if "operation" not in spec:
                    spec["operation"] = {
                        "train": "fresh", "score": True, "input_checkpoint_id": None,
                        "output_checkpoint_id": row["round_id"],
                    }
                    self._conn.execute(
                        "UPDATE rounds SET spec=? WHERE round_id=?",
                        (json.dumps(spec, sort_keys=True), row["round_id"]),
                    )
            for row in self._conn.execute("SELECT campaign_id, spec FROM campaigns").fetchall():
                spec = json.loads(row["spec"])
                if "train_mode" not in spec:
                    spec["train_mode"] = "fresh"
                    self._conn.execute(
                        "UPDATE campaigns SET spec=? WHERE campaign_id=?",
                        (json.dumps(spec, sort_keys=True), row["campaign_id"]),
                    )
        # Auth (node.secret, submissions.signature_ok) was dropped for the hackathon build. No
        # migration to drop the columns from an old DB — data/ is disposable; `rm -rf data/` instead.

    # One connection is shared by FastAPI's worker threads: every read goes through the lock too.
    def _all(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # --- nodes ------------------------------------------------------------------
    def register_node(self, node_id: str, name: str, specs: Dict[str, Any]) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO nodes (node_id, name, specs, registered_at, updated_at) VALUES (?,?,?,?,?)",
                (node_id, name, json.dumps(specs, sort_keys=True), now, now),
            )

    def update_node(self, node_id: str, name: str, specs: Dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE nodes SET name=?, specs=?, updated_at=? WHERE node_id=?",
                (name, json.dumps(specs, sort_keys=True), time.time(), node_id),
            )

    def record_heartbeat(self, node_id: str, heartbeat: Dict[str, Any], retention_s: float) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE nodes SET last_heartbeat=?, heartbeat=? WHERE node_id=?",
                (now, json.dumps(heartbeat, sort_keys=True), node_id),
            )
            self._conn.execute(
                """INSERT INTO heartbeat_samples (node_id, received_at, status, stage, round_id, load)
                   VALUES (?,?,?,?,?,?)""",
                (node_id, now, heartbeat.get("status", "idle"), heartbeat.get("stage"),
                 heartbeat.get("round_id"), json.dumps(heartbeat.get("load", {}), sort_keys=True)),
            )
            self._conn.execute("DELETE FROM heartbeat_samples WHERE received_at < ?", (now - retention_s,))

    def heartbeat_history(
        self,
        *,
        limit_per_node: int,
        since: Optional[float] = None,
        node_id: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Newest persisted heartbeat samples per node, returned oldest-first for graphing."""
        clauses, params = [], []
        if since is not None:
            clauses.append("received_at >= ?")
            params.append(since)
        if node_id is not None:
            clauses.append("node_id = ?")
            params.append(node_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit_per_node)
        rows = self._all(
            f"""
            SELECT node_id, received_at, status, stage, round_id, load
            FROM (
                SELECT node_id, received_at, status, stage, round_id, load,
                       ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY received_at DESC, id DESC) AS rn
                FROM heartbeat_samples
                {where}
            )
            WHERE rn <= ?
            ORDER BY node_id, received_at
            """,
            tuple(params),
        )
        history: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            sample = dict(row)
            sample["load"] = json.loads(sample["load"])
            history.setdefault(sample.pop("node_id"), []).append(sample)
        return history

    @staticmethod
    def _node(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["specs"] = _loads(d["specs"])
        d["heartbeat"] = _loads(d["heartbeat"])
        return d

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        row = self._one("SELECT * FROM nodes WHERE node_id=?", (node_id,))
        return self._node(row) if row else None

    def list_nodes(self) -> List[Dict[str, Any]]:
        return [self._node(r) for r in self._all("SELECT * FROM nodes ORDER BY registered_at")]

    # --- rounds -----------------------------------------------------------------
    def create_round(self, round_id: str, budget_k: int, note: Optional[str],
                     spec: Dict[str, Any], participants: List[Dict[str, str]]) -> bool:
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO rounds (round_id, budget_k, note, created_at, spec)"
                " VALUES (?,?,?,?,?)",
                (round_id, budget_k, note, now, json.dumps(spec, sort_keys=True)),
            )
            if cur.rowcount != 1:
                return False
            self._conn.executemany(
                "INSERT INTO participants (round_id, node_id, dataset_id, assigned_at, params) VALUES (?,?,?,?,?)",
                ((round_id, p["node_id"], p["dataset_id"], now, json.dumps(p.get("params") or {}, sort_keys=True))
                 for p in participants),
            )
            return True

    @staticmethod
    def _round(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["spec"] = _loads(d["spec"])
        if "participant_params" in d:
            d["participant_params"] = _loads(d["participant_params"]) or {}
        return d

    def get_round(self, round_id: str) -> Optional[Dict[str, Any]]:
        row = self._one("SELECT * FROM rounds WHERE round_id=?", (round_id,))
        return self._round(row) if row else None

    def list_rounds(self) -> List[Dict[str, Any]]:
        return [self._round(r) for r in self._all("SELECT * FROM rounds ORDER BY created_at")]

    def set_round_status(self, round_id: str, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE rounds SET status=? WHERE round_id=?", (status, round_id))

    # --- participants / tasks ------------------------------------------------------
    def participants_for_round(self, round_id: str) -> List[Dict[str, Any]]:
        rows = self._all(
            "SELECT node_id, dataset_id, status, assigned_at, accepted_at, submitted_at, params"
            " FROM participants WHERE round_id=? ORDER BY node_id", (round_id,))
        return [{**dict(r), "params": _loads(r["params"]) or {}} for r in rows]

    def get_task(self, node_id: str, round_id: str) -> Optional[Dict[str, Any]]:
        row = self._one(TASK_SELECT + " WHERE p.node_id=? AND p.round_id=?", (node_id, round_id))
        return self._round(row) if row else None

    def tasks_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """Open rounds this node participates in and has not submitted to yet."""
        rows = self._all(
            TASK_SELECT + " WHERE p.node_id=? AND r.status='open' AND p.status != 'submitted'"
            " ORDER BY p.assigned_at", (node_id,))
        return [self._round(r) for r in rows]

    def mark_accepted(self, node_id: str, round_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE participants SET status='accepted', accepted_at=?"
                " WHERE node_id=? AND round_id=? AND status='assigned'",
                (time.time(), node_id, round_id),
            )

    def mark_submitted(self, node_id: str, round_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE participants SET status='submitted', submitted_at=? WHERE node_id=? AND round_id=?",
                (time.time(), node_id, round_id),
            )

    # --- submissions ------------------------------------------------------------
    def upsert_submission(
        self,
        *,
        round_id: str,
        node_id: str,
        scores: List[Dict[str, Any]],
        agg_stats: Dict[str, Any],
        eval_spearman: Optional[float],
        trust: str,
        payload_sha256: str,
    ) -> Dict[str, Any]:
        """Re-submission for the same (round, node) replaces the previous one."""
        with self._lock, self._conn:
            prev = self._conn.execute(
                "SELECT id, revision FROM submissions WHERE round_id=? AND node_id=?",
                (round_id, node_id),
            ).fetchone()
            revision = (prev["revision"] + 1) if prev else 1
            if prev:
                self._conn.execute("DELETE FROM submissions WHERE id=?", (prev["id"],))
            cur = self._conn.execute(
                """INSERT INTO submissions (round_id, node_id, revision, received_at, n_scores,
                       eval_spearman, trust, agg_stats, payload_sha256)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (round_id, node_id, revision, time.time(), len(scores), eval_spearman, trust,
                 json.dumps(agg_stats, sort_keys=True), payload_sha256),
            )
            sub_id = cur.lastrowid
            self._conn.executemany(
                "INSERT INTO scores (submission_id, chunk_id, score) VALUES (?,?,?)",
                ((sub_id, s["chunk_id"], s["score"]) for s in scores),
            )
        return {"submission_id": sub_id, "revision": revision}

    def submissions_for_round(self, round_id: str) -> List[Dict[str, Any]]:
        rows = self._all("SELECT * FROM submissions WHERE round_id=? ORDER BY node_id", (round_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["agg_stats"] = json.loads(d["agg_stats"])
            out.append(d)
        return out

    def scores_for_submission(self, submission_id: int) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._all(
            "SELECT chunk_id, score FROM scores WHERE submission_id=?", (submission_id,))]

    # --- chunks / labels (server-master pool; see campaigns.py) -------------------------
    def add_chunks(self, shard_id: str, chunks: List[Dict[str, str]]) -> int:
        """Insert (chunk_id, text) pairs into a shard; existing chunk_ids are left untouched. Returns
        how many were newly inserted."""
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.executemany(
                "INSERT OR IGNORE INTO chunks (chunk_id, shard_id, text, added_at) VALUES (?,?,?,?)",
                ((c["chunk_id"], shard_id, c["text"], now) for c in chunks),
            )
            return cur.rowcount

    def import_chunks(
        self,
        shard_id: str,
        rows: Iterable[Tuple[str, str, Optional[int]]],
        *,
        label_source: str,
    ) -> Dict[str, int]:
        """Atomically import a streamed dataset, including optional oracle labels."""
        now = time.time()
        received = inserted = duplicates = labeled = 0
        with self._lock, self._conn:
            for chunk_id, text, label in rows:
                received += 1
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO chunks (chunk_id, shard_id, text, added_at) VALUES (?,?,?,?)",
                    (chunk_id, shard_id, text, now),
                )
                if cur.rowcount == 1:
                    inserted += 1
                else:
                    existing = self._conn.execute(
                        "SELECT shard_id, text FROM chunks WHERE chunk_id=?", (chunk_id,)
                    ).fetchone()
                    if existing["shard_id"] != shard_id:
                        raise ValueError(
                            f"chunk_id {chunk_id!r} already belongs to shard {existing['shard_id']!r}"
                        )
                    if existing["text"] != text:
                        raise ValueError(f"chunk_id {chunk_id!r} already exists with different text")
                    duplicates += 1
                if label is not None:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO labels (chunk_id, label, source, labeled_at) VALUES (?,?,?,?)",
                        (chunk_id, label, label_source, now),
                    )
                    labeled += 1
            total = self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE shard_id=?", (shard_id,)
            ).fetchone()[0]
            total_labels = self._conn.execute(
                """SELECT COUNT(*) FROM labels l JOIN chunks c ON c.chunk_id=l.chunk_id
                   WHERE c.shard_id=?""",
                (shard_id,),
            ).fetchone()[0]
        return {
            "received": received,
            "new": inserted,
            "duplicates": duplicates,
            "labeled": labeled,
            "total": total,
            "total_labels": total_labels,
        }

    def chunk_ids_for_shard(self, shard_id: str) -> List[str]:
        """Sorted, so campaign code can build aligned numpy arrays without persisting an index."""
        return [r["chunk_id"] for r in self._all(
            "SELECT chunk_id FROM chunks WHERE shard_id=? ORDER BY chunk_id", (shard_id,))]

    def chunk_texts(self, chunk_ids: List[str]) -> Dict[str, str]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._all(f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})", tuple(chunk_ids))
        return {r["chunk_id"]: r["text"] for r in rows}

    def add_labels(self, labels: Dict[str, int], source: str) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO labels (chunk_id, label, source, labeled_at) VALUES (?,?,?,?)",
                ((cid, label, source, now) for cid, label in labels.items()),
            )

    def labels_for_shard(self, shard_id: str) -> Dict[str, int]:
        rows = self._all(
            "SELECT l.chunk_id, l.label FROM labels l JOIN chunks c ON c.chunk_id = l.chunk_id"
            " WHERE c.shard_id=?", (shard_id,))
        return {r["chunk_id"]: r["label"] for r in rows}

    def list_shards(self) -> List[Dict[str, Any]]:
        """Dataset inventory without loading chunk text into memory."""
        return [dict(row) for row in self._all(
            """SELECT c.shard_id, COUNT(*) AS n_chunks, COUNT(l.chunk_id) AS n_labels,
                      MIN(c.added_at) AS created_at, MAX(c.added_at) AS updated_at
               FROM chunks c LEFT JOIN labels l ON l.chunk_id = c.chunk_id
               GROUP BY c.shard_id ORDER BY c.shard_id"""
        )]

    def shard_preview(self, shard_id: str, *, offset: int, limit: int,
                      query: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """One searchable page of chunks joined to their optional oracle label."""
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM chunks WHERE shard_id=? LIMIT 1", (shard_id,)
            ).fetchone()
            if not exists:
                return None
            clauses = ["c.shard_id = ?"]
            params: List[Any] = [shard_id]
            if query:
                clauses.append("(instr(lower(c.chunk_id), lower(?)) > 0 OR instr(lower(c.text), lower(?)) > 0)")
                params.extend([query, query])
            where = " AND ".join(clauses)
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM chunks c WHERE {where}", tuple(params)
            ).fetchone()[0]
            rows = self._conn.execute(
                f"""SELECT c.chunk_id, c.text, c.added_at, l.label, l.source, l.labeled_at
                    FROM chunks c LEFT JOIN labels l ON l.chunk_id = c.chunk_id
                    WHERE {where} ORDER BY c.chunk_id LIMIT ? OFFSET ?""",
                (*params, limit, offset),
            ).fetchall()
        return {"total": total, "chunks": [dict(row) for row in rows]}

    # --- campaigns (server-master active learning loop) ---------------------------------
    def create_campaign(self, campaign_id: str, shard_id: str, spec: Dict[str, Any]) -> bool:
        with self._lock, self._conn:
            now = time.time()
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO campaigns (campaign_id, shard_id, spec, created_at, updated_at)"
                " VALUES (?,?,?,?,?)",
                (campaign_id, shard_id, json.dumps(spec, sort_keys=True), now, now),
            )
            return cur.rowcount == 1

    @staticmethod
    def _campaign(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["spec"] = _loads(d["spec"])
        d["result"] = _loads(d["result"])
        return d

    def get_campaign(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        row = self._one("SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,))
        return self._campaign(row) if row else None

    def list_campaigns(self) -> List[Dict[str, Any]]:
        return [self._campaign(r) for r in self._all("SELECT * FROM campaigns ORDER BY created_at")]

    def list_shard_ids(self) -> List[str]:
        return [r["shard_id"] for r in self._all("SELECT DISTINCT shard_id FROM chunks ORDER BY shard_id")]

    def add_campaign_round(self, campaign_id: str, round_id: str, step: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO campaign_rounds (campaign_id, round_id, step) VALUES (?,?,?)",
                (campaign_id, round_id, step),
            )
            self._conn.execute(
                "UPDATE campaigns SET rounds_done=?, updated_at=? WHERE campaign_id=?",
                (step, time.time(), campaign_id),
            )

    def current_campaign_round(self, campaign_id: str) -> Optional[str]:
        row = self._one(
            "SELECT round_id FROM campaign_rounds WHERE campaign_id=? ORDER BY step DESC LIMIT 1",
            (campaign_id,))
        return row["round_id"] if row else None

    def finish_campaign(self, campaign_id: str, result: Dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE campaigns SET status='done', result=?, updated_at=? WHERE campaign_id=?",
                (json.dumps(result), time.time(), campaign_id),
            )
