"""Data carried between the server, the trainer and the scorer."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

LABEL_PREFIX = "__label__"


@dataclass(frozen=True)
class Sample:
    """An unlabeled document to score (rounds 2 and 3)."""

    sample_id: str
    text: str


@dataclass(frozen=True)
class LabeledSample:
    """A golden document carrying the teacher LLM's score (round 1)."""

    sample_id: str
    text: str
    label: str


@dataclass(frozen=True)
class ScoredSample:
    """What the node sends back for one document.

    The full distribution is returned on purpose: active learning needs the
    classifier's uncertainty to decide what to label next, and an argmax label
    alone throws that away.
    """

    sample_id: str
    label: str
    probs: dict[str, float]
    entropy: float
    margin: float
    expected_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """fastText hyperparameters plus the knobs governing how we pick them."""

    lr: float = 0.5
    # None derives the epoch count from the dataset size. fastText's stock
    # default is tuned for large corpora and badly underfits small ones: on the
    # 260-sample smoke set, 25 epochs scores 0.20 accuracy while 300 scores
    # 0.975. Since the data volume is unknown up front, what stays roughly
    # constant is the update budget, not the number of passes.
    epoch: int | None = None
    target_updates: int = 200_000
    min_epoch: int = 5
    max_epoch: int = 300

    word_ngrams: int = 2
    dim: int = 100
    min_count: int = 1
    loss: str = "softmax"

    # None sizes the n-gram hashing table against the corpus. fastText's stock
    # 2_000_000 buckets cost dim * 4 bytes each -- an 800 MB model for a 300
    # document dataset -- which is absurd for a node that ships the model back
    # and forth between rounds.
    bucket: int | None = None
    tokens_per_bucket: int = 5
    min_bucket: int = 10_000
    max_bucket: int = 2_000_000
    # 0 means "one per core", resolved by resolve_thread(). It must never
    # reach fastText: passing thread=0 divides work by the thread count and
    # kills the process with SIGFPE, deterministically. Tests set 1, because
    # training is otherwise non-reproducible (asynchronous SGD across threads).
    thread: int = 0
    # fastText prints progress to stderr at its default verbosity, which is
    # noise for a library. The CLI raises it.
    verbose: int = 0

    # Autotune needs a validation split to search against, which only pays off
    # once there is enough data to spare. Below the threshold we use the
    # defaults above.
    autotune_seconds: int = 60
    autotune_min_samples: int = 2000
    # e.g. "50M". Autotune optimizes validation score alone and will happily
    # produce a 262 MB model for 2500 documents; this caps it. fastText hits
    # the target by quantizing, so the saved artifact becomes a .ftz.
    autotune_model_size: str | None = None

    holdout_fraction: float = 0.15
    split_seed: int = 17

    # fastText intermittently aborts training with "Encountered NaN" -- a
    # stochastic upstream defect, not a configuration problem: it survives a
    # build from C++ sources, and the same parameters succeed on a retry
    # because the weight initialization is random. Losing a node's task to it
    # would be worse than spending a few seconds again.
    train_attempts: int = 3

    # Train in a freshly spawned process. fastText accumulates state across
    # trainings in a long-lived interpreter and eventually aborts with
    # "Encountered NaN" -- not reproducible in a fresh process, and not fixed
    # by building from C++ sources. A new interpreter per training sidesteps it
    # entirely. Turn off only for tests that patch fastText in-process.
    isolate_training: bool = True

    quantize: bool = False
    # fastText can retrain while quantizing, but on a small corpus with a high
    # epoch count that optimization diverges outright ("Encountered NaN").
    # It only buys anything when feature pruning is on, which it is not here.
    quantize_retrain: bool = False
    # Metrics are measured on a held-out slice, then the shipped model is
    # retrained on everything. Standard practice, but it means the report
    # describes the procedure rather than the exact weights shipped.
    retrain_on_full: bool = True

    def resolve_thread(self) -> int:
        """Never returns 0 -- see the note on `thread`."""
        if self.thread > 0:
            return self.thread
        return max(1, os.cpu_count() or 1)

    def resolve_epoch(self, n_samples: int) -> int:
        """Pick an epoch count that keeps the total update budget roughly fixed."""
        if self.epoch is not None:
            return self.epoch
        if n_samples <= 0:
            return self.min_epoch
        derived = self.target_updates // n_samples
        return int(max(self.min_epoch, min(self.max_epoch, derived)))

    def resolve_bucket(self, n_tokens: int) -> int:
        """Size the hashing table to the corpus, not to a fixed constant."""
        if self.bucket is not None:
            return self.bucket
        derived = self.tokens_per_bucket * max(n_tokens, 0)
        return int(max(self.min_bucket, min(self.max_bucket, derived)))

    def fasttext_kwargs(self, n_samples: int, n_tokens: int) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "epoch": self.resolve_epoch(n_samples),
            "wordNgrams": self.word_ngrams,
            "dim": self.dim,
            "minCount": self.min_count,
            "loss": self.loss,
            "bucket": self.resolve_bucket(n_tokens),
            "thread": self.resolve_thread(),
        }


@dataclass(frozen=True)
class ClassReport:
    label: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class TrainReport:
    """Everything the server needs to judge how much to trust the scores."""

    n_samples: int
    n_skipped: int
    n_train: int
    n_eval: int
    labels: list[str]
    autotuned: bool

    # All four are None when the dataset was too small to hold anything out,
    # which is honester than reporting an accuracy of 0.0 for a model that was
    # simply never measured.
    accuracy: float | None
    macro_f1: float | None
    # Ordinal metrics, populated only when every label parses as a number.
    # Accuracy alone would punish an off-by-one the same as an off-by-seven.
    mae: float | None
    qwk: float | None

    per_class: list[ClassReport] = field(default_factory=list)
    # Non-fatal problems worth passing back to the server, e.g. quantization
    # that was requested but could not be applied.
    warnings: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    metrics_measured_on: str = "holdout"
    model_trained_on: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
