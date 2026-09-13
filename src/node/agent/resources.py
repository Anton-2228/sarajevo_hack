"""Turning "I give you 2 cores and 4 GB" into something fastText obeys.

Three levers, in increasing order of violence:

1. *Shape the work to fit.* The n-gram table is sized against the RAM budget,
   so the model the round produces is one the node can actually hold. This is
   the lever that should do all the work.
2. *Constrain the machinery.* `thread`, the OpenMP variables and a CPU affinity
   mask, so the agent uses the cores it was given and no others.
3. *Cap it hard.* An address-space limit inside the training child, applied by
   `node.core.limits`. If levers 1 and 2 were wrong, this one is what keeps the
   promise -- by killing the training rather than the machine.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import psutil

from node.agent.models import GPU
from node.agent.reporting import Reporter
from node.core.limits import ENV_CORE_IDS, ENV_RAM_BYTES, ENV_THREADS, THREAD_ENV_VARS
from node.core.types import TrainConfig

# The parent holds things the child does not: the whole scored corpus and the
# JSON body on its way out. The child gets most of the budget, not all of it.
TRAIN_SHARE = 0.80

# The hashing table is not the only resident structure -- there is also the
# output matrix, the vocabulary, per-thread state, and with retrain_on_full a
# second training in the same process. A quarter of the budget for the table
# leaves room for the rest.
BUCKET_SHARE = 0.25
BYTES_PER_BUCKET_ENTRY = 4  # float32

# fastText's own default is 0.5 and the useful range is roughly 0.05-1.0. A
# round that asks for a neural-sized learning rate would otherwise produce a
# model that never trained, and report it as a successful round.
MIN_USABLE_LR = 0.05
MAX_USABLE_LR = 2.0


@dataclass(frozen=True)
class HardwareInfo:
    logical_cores: int
    total_ram_gb: float
    disk_free_gb: float | None
    gpus: list[GPU] = field(default_factory=list)
    platform: str = sys.platform
    python: str = ""


def detect_hardware(probe_path: Path | None = None) -> HardwareInfo:
    """What this machine has, for the handshake.

    `gpus` is always empty. The node's only model kind is a fastText
    classifier; advertising a GPU would invite LoRA rounds it cannot bake.
    """
    try:
        free = shutil.disk_usage(probe_path or Path.cwd()).free / 1024**3
    except OSError:
        free = None

    return HardwareInfo(
        logical_cores=psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        total_ram_gb=psutil.virtual_memory().total / 1024**3,
        disk_free_gb=round(free, 2) if free is not None else None,
        gpus=[],
        platform=sys.platform,
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
    )


@dataclass(frozen=True)
class Budget:
    cores: int
    ram_bytes: int
    core_ids: list[int] = field(default_factory=list)

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / 1024**3

    def train_ram_bytes(self) -> int:
        return int(self.ram_bytes * TRAIN_SHARE)


def resolve_budget(
    cores: int, ram_gb: float, hw: HardwareInfo
) -> tuple[Budget, list[str]]:
    """Clamp the request to something workable and say what was adjusted."""
    warnings: list[str] = []

    if cores < 1:
        cores = max(1, hw.logical_cores - 1)
    if cores > hw.logical_cores:
        warnings.append(
            f"asked for {cores} cores but the machine has {hw.logical_cores}; "
            "the agent will oversubscribe"
        )

    if ram_gb <= 0:
        ram_gb = min(4.0, hw.total_ram_gb * 0.5)
    if ram_gb > hw.total_ram_gb:
        warnings.append(
            f"asked for {ram_gb:.1f} GiB but the machine has {hw.total_ram_gb:.1f} GiB; "
            "training will hit the cap before the machine runs out"
        )

    core_ids = pick_core_ids(cores)
    if len(core_ids) < cores:
        warnings.append(
            f"only {len(core_ids)} cores are available to this process, not {cores}"
        )

    return Budget(cores=cores, ram_bytes=int(ram_gb * 1024**3), core_ids=core_ids), warnings


def pick_core_ids(n: int) -> list[int]:
    """Which logical cores to pin to.

    On Linux this starts from the set we are already allowed to use, so the
    agent composes correctly inside a cpuset or a container instead of asking
    for cores that were never ours.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(os.sched_getaffinity(0))[:n]
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            return _windows_core_ids()[:n]
        except OSError:
            pass
    return list(range(n))


def _windows_core_ids() -> list[int]:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessAffinityMask.restype = wintypes.BOOL

    process_mask = ctypes.c_size_t(0)
    system_mask = ctypes.c_size_t(0)
    if not kernel32.GetProcessAffinityMask(
        kernel32.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask failed")
    return [i for i in range(64) if process_mask.value & (1 << i)]


def max_bucket_for(ram_bytes: int, dim: int, base: TrainConfig | None = None) -> int:
    """Size the n-gram table so the model fits the budget instead of busting it."""
    base = base or TrainConfig()
    derived = int(ram_bytes * BUCKET_SHARE / max(dim * BYTES_PER_BUCKET_ENTRY, 1))
    return int(max(base.min_bucket, min(base.max_bucket, derived)))


def train_config_for(
    budget: Budget,
    params: dict[str, float],
    base: TrainConfig | None = None,
    *,
    verbose: int = 0,
) -> tuple[TrainConfig, list[str], dict[str, float]]:
    """Fold the round's hyperparameters into a config the budget can afford.

    Returns the config, any warnings, and the parameters we actually used --
    the third is reported back to the server, because a node that quietly
    ignores an instruction is worse than one that says it did.
    """
    base = base or TrainConfig()
    warnings: list[str] = []
    effective: dict[str, float] = {}

    lr = base.lr
    if "proxy_lr" in params:
        requested = params["proxy_lr"]
        lr = min(max(requested, MIN_USABLE_LR), MAX_USABLE_LR)
        if lr != requested:
            warnings.append(
                f"proxy_lr={requested:g} is outside fastText's usable range; "
                f"training with lr={lr:g}"
            )
        effective["proxy_lr_effective"] = lr

    epoch = base.epoch
    if "epoch" in params:
        epoch = int(max(1, min(params["epoch"], base.max_epoch)))
        effective["epoch"] = float(epoch)

    dim = base.dim
    if "dim" in params:
        dim = int(min(max(params["dim"], 10), 300))
        effective["dim"] = float(dim)

    word_ngrams = base.word_ngrams
    for key in ("word_ngrams", "wordNgrams"):
        if key in params:
            word_ngrams = int(min(max(params[key], 1), 5))
            effective["word_ngrams"] = float(word_ngrams)
            break

    min_count = base.min_count
    if "min_count" in params:
        min_count = int(max(1, params["min_count"]))
        effective["min_count"] = float(min_count)

    split_seed = base.split_seed
    if "seed" in params:
        split_seed = int(params["seed"])

    config = replace(
        base,
        lr=lr,
        epoch=epoch,
        dim=dim,
        word_ngrams=word_ngrams,
        min_count=min_count,
        split_seed=split_seed,
        # Never 0: a zero thread count reaching fastText divides by it and
        # kills the process with SIGFPE, which is not an exception you can
        # catch.
        thread=max(1, budget.cores),
        bucket=None,
        max_bucket=max_bucket_for(budget.train_ram_bytes(), dim, base),
        verbose=verbose,
        # Both the NaN mitigation and the only place the memory cap can apply.
        isolate_training=True,
    )
    return config, warnings, effective


def child_env(budget: Budget, *, enforce: bool = True) -> dict[str, str]:
    """The budget, in the form the training child reads it."""
    env = {name: str(budget.cores) for name in THREAD_ENV_VARS}
    env[ENV_THREADS] = str(budget.cores)
    if enforce:
        env[ENV_RAM_BYTES] = str(budget.train_ram_bytes())
        if budget.core_ids:
            env[ENV_CORE_IDS] = ",".join(str(c) for c in budget.core_ids)
    return env


@contextmanager
def applied_env(env: dict[str, str]) -> Iterator[None]:
    """Set variables for the duration of a block, then put things back exactly.

    The agent is long-lived and trains many rounds. A leaked OMP_NUM_THREADS
    would silently constrain every later one, which is the kind of bug that
    only shows up as "the node got slower and nobody knows when".
    """
    previous = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def current_load(budget: Budget) -> dict[str, float]:
    """Numbers for the heartbeat. The schema accepts numerics and nothing else."""
    process = psutil.Process()
    try:
        rss = process.memory_info().rss
        tree = rss + sum(c.memory_info().rss for c in process.children(recursive=True))
    except (psutil.Error, OSError):
        rss = tree = 0

    virtual = psutil.virtual_memory()
    return {
        "cpu_pct": float(psutil.cpu_percent(interval=None)),
        "ram_pct": float(virtual.percent),
        "agent_rss_mb": round(rss / 1024**2, 1),
        "tree_rss_mb": round(tree / 1024**2, 1),
        "budget_ram_pct": round(100.0 * tree / budget.ram_bytes, 1) if budget.ram_bytes else 0.0,
    }


def prime_cpu_percent() -> None:
    """psutil's first cpu_percent call always returns 0.0; spend it early."""
    psutil.cpu_percent(interval=None)


class MemoryWatchdog:
    """Watches the whole process tree's memory while a round runs.

    It reports; it does not kill. The hard cap inside the training child is
    what enforces the budget. This exists because the child's pid never reaches
    us -- it is spawned deep inside `classifier._run_isolated` -- so walking our
    own tree is the only handle we have on what it is doing, and `peak_rss_mb`
    is genuinely useful for the scheduler sizing future rounds.
    """

    def __init__(self, budget: Budget, reporter: Reporter, interval_s: float = 0.5) -> None:
        self._budget = budget
        self._reporter = reporter
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_bytes = 0
        self.breached = False

    def __enter__(self) -> MemoryWatchdog:
        self._thread = threading.Thread(target=self._watch, name="memwatch", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)

    @property
    def peak_rss_mb(self) -> float | None:
        """None when nothing was ever sampled -- not the same as zero bytes."""
        if self.peak_rss_bytes == 0:
            return None
        return round(self.peak_rss_bytes / 1024**2, 1)

    def _watch(self) -> None:
        import logging

        warned = False
        process = psutil.Process()
        first = True
        while first or not self._stop.wait(self._interval):
            # Sample immediately, then on the interval: a training run shorter
            # than one tick would otherwise report a peak of zero, which reads
            # as a measurement rather than the absence of one.
            first = False
            try:
                total = process.memory_info().rss + sum(
                    c.memory_info().rss for c in process.children(recursive=True)
                )
            except (psutil.Error, OSError):
                continue

            self.peak_rss_bytes = max(self.peak_rss_bytes, total)
            if total > self._budget.ram_bytes * 0.9:
                self.breached = True
                if not warned:
                    warned = True
                    self._reporter.note(
                        logging.WARNING,
                        "memory use is approaching the budget",
                        used_mb=round(total / 1024**2, 1),
                        budget_mb=round(self._budget.ram_bytes / 1024**2, 1),
                    )


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Min/max/mean of a score column, for a one-line log that means something."""
    if not values:
        return {}
    return {
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(sum(values) / len(values), 4),
    }
