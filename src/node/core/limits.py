"""Resource caps a training process applies to itself.

A node advertises how much of the machine it is willing to spend, and the
promise has to be kept even when a round arrives with hyperparameters that
would happily allocate the whole box. Enforcing that from the outside is
awkward: the training process is spawned three frames deep inside
`classifier._run_isolated`, and its pid never reaches the caller. So the child
caps itself instead, reading a budget the parent left in the environment --
`_run_isolated` runs `subprocess.run` without `env=`, so the child inherits it.

The caps are deliberately hard. When one is hit the allocation fails, C++
`new` throws, the worker dies without writing its result file, and the existing
`_run_isolated` turns that into

    RuntimeError("training worker exited with N and produced nothing")

which is the same signature on Linux and on Windows. One error to handle, one
meaning: the round wanted more than the node promised.

Nothing here raises. A cap that cannot be applied is reported in
`AppliedLimits.failures` and the training runs anyway -- a node that refuses to
work because it could not install a safety net is worse than one that works
without it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

ENV_RAM_BYTES = "NODE_LIMIT_RAM_BYTES"
ENV_CORE_IDS = "NODE_LIMIT_CORE_IDS"
ENV_THREADS = "NODE_LIMIT_THREADS"

# Every library in the stack reads its own variable, and fastText's OpenMP
# runtime reads the first one at load time -- which is why this must be set
# before numpy or fasttext is imported, not after.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

# The job handle has to outlive this call: closing it drops the limit with it.
_job_handle: int | None = None


@dataclass(frozen=True)
class AppliedLimits:
    """What the process actually managed to impose on itself."""

    ram_bytes: int | None = None
    core_ids: list[int] = field(default_factory=list)
    threads: int | None = None
    applied: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def apply_from_env(environ: dict[str, str] | None = None) -> AppliedLimits:
    """Read the budget from the environment and impose it. Never raises."""
    env = os.environ if environ is None else environ

    ram_bytes = _read_int(env, ENV_RAM_BYTES)
    threads = _read_int(env, ENV_THREADS)
    core_ids = _read_core_ids(env.get(ENV_CORE_IDS))

    applied: list[str] = []
    failures: list[str] = []

    def record(result: tuple[str | None, str | None]) -> None:
        note, problem = result
        if note:
            applied.append(note)
        if problem:
            failures.append(problem)

    # Threads first: the OpenMP runtime samples its variable when it loads, so
    # setting it after fastText is imported changes nothing at all.
    if threads:
        record(set_thread_env(threads))
    if core_ids:
        record(set_affinity(core_ids))
    if ram_bytes:
        record(limit_memory(ram_bytes))

    return AppliedLimits(
        ram_bytes=ram_bytes,
        core_ids=core_ids,
        threads=threads,
        applied=applied,
        failures=failures,
    )


def set_thread_env(n: int) -> tuple[str | None, str | None]:
    """Cap every threading runtime we might pull in. Returns (applied, failure)."""
    if n < 1:
        return None, f"ignored thread count {n}"
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(n)
    return f"threads={n}", None


def set_affinity(core_ids: list[int]) -> tuple[str | None, str | None]:
    """Pin this process to the given logical cores."""
    if not core_ids:
        return None, None
    if sys.platform == "win32":
        return _set_affinity_windows(core_ids)
    # macOS has no affinity API at all, so probe the capability rather than
    # the platform name.
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(core_ids))
        except OSError as error:
            return None, f"affinity: {error}"
        return f"affinity={_summarize(core_ids)}", None
    return None, "affinity: unsupported on this platform"


def limit_memory(nbytes: int) -> tuple[str | None, str | None]:
    """Impose a hard ceiling on this process's memory."""
    if nbytes < 1:
        return None, f"ignored memory limit {nbytes}"
    if sys.platform == "win32":
        return _limit_memory_windows(nbytes)
    return _limit_memory_posix(nbytes)


def _limit_memory_posix(nbytes: int) -> tuple[str | None, str | None]:
    """RLIMIT_AS -- address space, which is a slight overestimate of usage.

    Virtual size exceeds resident size, so this cap bites a little sooner than
    a pure RSS limit would. That is the conservative direction: promising 2 GB
    and using 1.9 is fine, promising 2 and using 3 is what we must prevent.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - POSIX always has it
        return None, "memory: `resource` unavailable"

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # Never try to raise the ceiling: an unprivileged process cannot, and
        # the attempt fails the whole call.
        target = nbytes if hard == resource.RLIM_INFINITY else min(nbytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    except (OSError, ValueError) as error:
        return None, f"memory: {error}"

    note = f"RLIMIT_AS={_gib(target)}"
    if sys.platform == "darwin":
        # Reported as applied but flagged: macOS accepts the call and then does
        # not reliably honour it.
        return note, "memory: RLIMIT_AS is advisory on macOS"
    return note, None


def _limit_memory_windows(nbytes: int) -> tuple[str | None, str | None]:
    """A job object with a per-process memory cap.

    Windows 8 and later allow nested jobs, so this still works when the process
    is already inside one (a container, or a CI runner).
    """
    global _job_handle

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    try:
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None, f"memory: CreateJobObject failed ({ctypes.get_last_error()})"

        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = ctypes.c_size_t(nbytes).value

        ok = kernel32.SetInformationJobObject(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            return None, f"memory: SetInformationJobObject failed ({ctypes.get_last_error()})"

        # Read the limit back. A wrong struct layout does not fail the call --
        # it writes the number into a neighbouring field and silently enforces
        # nothing, which is the worst possible outcome for a safety cap.
        check = _ExtendedLimits()
        returned = wintypes.DWORD(0)
        if kernel32.QueryInformationJobObject(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(check),
            ctypes.sizeof(check),
            ctypes.byref(returned),
        ):
            if check.ProcessMemoryLimit != nbytes:
                return None, (
                    "memory: job object read back "
                    f"{check.ProcessMemoryLimit} instead of {nbytes}"
                )

        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            return None, f"memory: AssignProcessToJobObject failed ({ctypes.get_last_error()})"
    except OSError as error:
        return None, f"memory: {error}"

    # Outlive this frame, or the handle closes and takes the limit with it.
    _job_handle = job
    return f"JobObject.ProcessMemoryLimit={_gib(nbytes)}", None


def _set_affinity_windows(core_ids: list[int]) -> tuple[str | None, str | None]:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    mask = 0
    for core in core_ids:
        # Beyond 64 logical processors Windows splits CPUs into groups and this
        # mask only addresses the current one. Cores we cannot name are dropped
        # rather than wrapping onto the wrong CPU.
        if 0 <= core < 64:
            mask |= 1 << core
    if not mask:
        return None, f"affinity: no addressable cores in {_summarize(core_ids)}"

    try:
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
            return None, f"affinity: SetProcessAffinityMask failed ({ctypes.get_last_error()})"
    except OSError as error:
        return None, f"affinity: {error}"
    return f"affinity={_summarize(core_ids)}", None


def _read_int(env: dict[str, str], name: str) -> int | None:
    raw = env.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_core_ids(raw: str | None) -> list[int]:
    if not raw:
        return []
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


def _summarize(core_ids: list[int]) -> str:
    return ",".join(str(c) for c in core_ids)


def _gib(nbytes: int) -> str:
    return f"{nbytes / (1024**3):.2f}GiB"
