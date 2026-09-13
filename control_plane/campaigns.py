"""Sharded and mixture-of-experts campaigns over a server-held chunk pool.

In ``sharded`` mode each node trains and scores only its non-overlapping partition, providing
throughput. In ``experts`` mode each node trains on its own domain partition but scores the whole
pool, providing multiple signals per chunk. Nodes still use the regular task/ack/submit transport.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from . import acquisition
from .oracle import Oracle
from .partitioning import chunk_ids_for_partition
from .pooling import NodeSubmission, select_topk
from .storage import Store


class CampaignError(ValueError):
    """Bad input or state transition; app.py turns this into a 4xx response."""


@dataclass
class AdvanceResult:
    status: str  # waiting | advanced | done
    detail: str
    round_id: Optional[str] = None


class CampaignService:
    def __init__(self, store: Store, oracle: Oracle):
        self.store = store
        self.oracle = oracle

    def create_sharded(self, campaign_id: str, shard_id: str, model: Dict[str, str], metrics: List[str],
                       node_ids: List[str], schedule: List[int], train_mode: str, strategy: str, k_frac: float,
                       good_min: Optional[int], seed: int) -> Dict[str, Any]:
        return self._create_common("sharded", campaign_id, shard_id, model, metrics, node_ids,
                                   schedule, train_mode, strategy, k_frac, good_min, seed)

    def create_experts(self, campaign_id: str, shard_id: str, model: Dict[str, str], metrics: List[str],
                       node_ids: List[str], schedule: List[int], train_mode: str, strategy: str, k_frac: float,
                       good_min: Optional[int], seed: int) -> Dict[str, Any]:
        return self._create_common("experts", campaign_id, shard_id, model, metrics, node_ids,
                                   schedule, train_mode, strategy, k_frac, good_min, seed)

    def _create_common(self, mode: str, campaign_id: str, shard_id: str, model: Dict[str, str],
                       metrics: List[str], node_ids: List[str], schedule: List[int], train_mode: str,
                       strategy: str, k_frac: float, good_min: Optional[int], seed: int) -> Dict[str, Any]:
        pool = self.store.chunk_ids_for_shard(shard_id)
        n_partitions = len(node_ids)
        partitions = [chunk_ids_for_partition(pool, i, n_partitions) for i in range(n_partitions)]
        undersized = [(i, len(part)) for i, part in enumerate(partitions) if len(part) < schedule[-1]]
        if undersized:
            partition, size = undersized[0]
            raise CampaignError(
                f"partition {partition} of shard {shard_id!r} has {size} chunks, "
                f"schedule needs {schedule[-1]} per partition"
            )

        for node_id in node_ids:
            node = self.store.get_node(node_id)
            if not node:
                raise CampaignError(f"participant {node_id!r} is not a registered node")
            if model["kind"] not in node["specs"]["model_kinds"]:
                raise CampaignError(f"node {node_id!r} cannot bake model kind {model['kind']!r}")

        spec = {
            "mode": mode,
            "n_partitions": n_partitions,
            "model": model,
            "metrics": metrics,
            "node_ids": node_ids,
            "schedule": schedule,
            "train_mode": train_mode,
            "strategy": strategy,
            "k_frac": k_frac,
            "good_min": good_min,
            "seed": seed,
        }
        if not self.store.create_campaign(campaign_id, shard_id, spec):
            raise CampaignError(f"campaign {campaign_id!r} already exists")

        existing_labels = self.store.labels_for_shard(shard_id)
        seed_ids: List[str] = []
        for partition, domain_ids in enumerate(partitions):
            free = [chunk_id for chunk_id in domain_ids if chunk_id not in existing_labels]
            needed = max(0, schedule[0] - (len(domain_ids) - len(free)))
            if needed:
                rng = np.random.default_rng([seed, 0, partition])
                seed_idx = rng.choice(len(free), needed, replace=False)
                seed_ids.extend(free[int(i)] for i in seed_idx)
        self._label(seed_ids)
        self._create_round(campaign_id, shard_id, spec, step=1)
        return self.store.get_campaign(campaign_id)

    def _label(self, chunk_ids: List[str]) -> None:
        if not chunk_ids:
            return
        texts = self.store.chunk_texts(chunk_ids)
        self.store.add_labels(self.oracle.label(chunk_ids, texts), source=self.oracle.name)

    def _create_round(self, campaign_id: str, shard_id: str, spec: Dict[str, Any], step: int) -> str:
        round_id = f"{campaign_id}-r{step}"
        n_pool = len(self.store.chunk_ids_for_shard(shard_id))
        budget_k = max(1, round(n_pool * spec["k_frac"]))
        participants = [
            {
                "node_id": node_id,
                "dataset_id": shard_id,
                "params": {
                    "partition": partition,
                    "n_partitions": spec["n_partitions"],
                    "mode": spec["mode"],
                },
            }
            for partition, node_id in enumerate(spec["node_ids"])
        ]
        requested_train_mode = spec.get("train_mode", "fresh")
        train = "continue" if requested_train_mode == "continue" and step > 1 else "fresh"
        round_spec = {
            "model": spec["model"],
            "operation": {
                "train": train,
                "score": True,
                "input_checkpoint_id": f"{campaign_id}-r{step - 1}" if train == "continue" else None,
                "output_checkpoint_id": round_id,
            },
            "metrics": spec["metrics"],
            "params": {"n_labels": spec["schedule"][step - 1]},
        }
        if not self.store.create_round(
            round_id, budget_k, f"campaign {campaign_id} step {step}", round_spec, participants
        ):
            raise CampaignError(f"round {round_id!r} already exists")  # pragma: no cover
        self.store.add_campaign_round(campaign_id, round_id, step)
        return round_id

    def advance(self, campaign_id: str) -> AdvanceResult:
        campaign = self.store.get_campaign(campaign_id)
        if not campaign:
            raise CampaignError(f"no such campaign {campaign_id!r}")
        if campaign["status"] == "done":
            return AdvanceResult("done", "campaign already finished")

        spec = campaign["spec"]
        mode = spec["mode"]
        n_partitions = spec["n_partitions"]
        round_id = self.store.current_campaign_round(campaign_id)
        submissions = {s["node_id"]: s for s in self.store.submissions_for_round(round_id)}
        missing = [node_id for node_id in spec["node_ids"] if node_id not in submissions]
        if missing:
            return AdvanceResult("waiting", f"waiting on {missing}", round_id=round_id)

        pool = self.store.chunk_ids_for_shard(campaign["shard_id"])
        per_node_scores: Dict[str, Dict[str, float]] = {}
        for partition, node_id in enumerate(spec["node_ids"]):
            rows = self.store.scores_for_submission(submissions[node_id]["id"])
            got = {row["chunk_id"]: row["score"] for row in rows}
            expected = (chunk_ids_for_partition(pool, partition, n_partitions)
                        if mode == "sharded" else pool)
            expected_set = set(expected)
            absent = [chunk_id for chunk_id in expected if chunk_id not in got]
            if absent:
                raise CampaignError(
                    f"node {node_id!r} did not score {len(absent)} expected chunks, e.g. {absent[:3]}"
                )
            unexpected = [chunk_id for chunk_id in got if chunk_id not in expected_set]
            if unexpected:
                scope = "its partition" if mode == "sharded" else "the campaign pool"
                raise CampaignError(
                    f"node {node_id!r} scored {len(unexpected)} chunks outside {scope}, e.g. {unexpected[:3]}"
                )
            per_node_scores[node_id] = {chunk_id: got[chunk_id] for chunk_id in expected}

        self.store.set_round_status(round_id, "closed")
        labels = self.store.labels_for_shard(campaign["shard_id"])
        step = campaign["rounds_done"]
        schedule = spec["schedule"]
        if step >= len(schedule):
            result = self._finalize(mode, pool, per_node_scores, spec, labels)
            self.store.finish_campaign(campaign_id, result)
            return AdvanceResult(
                "done", f"finished after {step} rounds, {len(labels)} labels", round_id=round_id
            )

        new_ids: List[str] = []
        target = schedule[step]
        for partition, node_id in enumerate(spec["node_ids"]):
            domain_ids = chunk_ids_for_partition(pool, partition, n_partitions)
            domain_index = {chunk_id: i for i, chunk_id in enumerate(domain_ids)}
            labeled_idx = np.array(
                [domain_index[chunk_id] for chunk_id in labels if chunk_id in domain_index], dtype=int
            )
            budget = max(0, target - len(labeled_idx))
            domain_scores = np.array([per_node_scores[node_id][chunk_id] for chunk_id in domain_ids])
            rng = np.random.default_rng([spec["seed"], step, partition])
            acquired = acquisition.acquire(
                spec["strategy"], budget, labeled_idx, domain_scores,
                np.zeros(len(domain_ids)), spec["k_frac"], rng,
            )
            new_ids.extend(domain_ids[int(i)] for i in acquired)

        self._label(new_ids)
        next_round = self._create_round(campaign_id, campaign["shard_id"], spec, step=step + 1)
        return AdvanceResult(
            "advanced", f"labelled {len(new_ids)} more ({target} per partition)", round_id=next_round
        )

    def _finalize(self, mode: str, pool: List[str], per_node_scores: Dict[str, Dict[str, float]],
                  spec: Dict[str, Any], labels: Dict[str, int]) -> Dict[str, Any]:
        """Put known-good chunks first, then fill from the mode-specific merged ranking."""
        k = max(1, round(len(pool) * spec["k_frac"]))
        good_min = spec["good_min"]
        known_good = (
            [chunk_id for chunk_id in pool if labels.get(chunk_id, good_min - 1) >= good_min]
            if good_min is not None else []
        )[:k]

        if mode == "sharded":
            nodes = [
                NodeSubmission(node_id, list(scores), list(scores.values()), curve=None)
                for node_id, scores in per_node_scores.items()
            ]
            merged = select_topk(
                nodes, len(pool), normalize="zscore", enforce_gate=False, include_unknown=True
            )["selected"]
            ranked_unlabeled = [item["chunk_id"] for item in merged if item["chunk_id"] not in labels]
        else:
            means = {
                chunk_id: statistics.fmean(scores[chunk_id] for scores in per_node_scores.values())
                for chunk_id in pool
            }
            ranked_unlabeled = sorted(
                (chunk_id for chunk_id in pool if chunk_id not in labels), key=lambda c: -means[c]
            )

        fill = ranked_unlabeled[:max(k - len(known_good), 0)]
        return {
            "selected": known_good + fill,
            "n_labels": len(labels),
            "n_known_good": len(known_good),
            "generated_at": time.time(),
        }
