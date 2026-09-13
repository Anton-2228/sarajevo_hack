"""Self-imposed resource caps.

The cap that matters is the one that actually bites, so the platform-specific
tests here do not assert that a call returned success -- they assert that an
over-budget allocation really fails.
"""

import subprocess
import sys

import pytest

from node.core import limits


def test_no_env_means_no_limits():
    applied = limits.apply_from_env({})
    assert applied.ram_bytes is None
    assert applied.core_ids == []
    assert applied.threads is None
    assert applied.applied == []
    assert applied.failures == []


def test_reads_the_budget_from_env(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    applied = limits.apply_from_env(
        {limits.ENV_THREADS: "3", limits.ENV_CORE_IDS: "0,1,2"}
    )
    assert applied.threads == 3
    assert applied.core_ids == [0, 1, 2]


def test_thread_env_reaches_every_runtime(monkeypatch):
    for name in limits.THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    note, problem = limits.set_thread_env(2)

    assert problem is None
    assert note == "threads=2"
    import os

    for name in limits.THREAD_ENV_VARS:
        assert os.environ[name] == "2"


def test_garbage_env_is_ignored_rather_than_fatal():
    # A malformed budget must not stop a node from working. It just means no
    # cap, which apply_from_env reports by leaving the fields empty.
    applied = limits.apply_from_env(
        {limits.ENV_RAM_BYTES: "lots", limits.ENV_CORE_IDS: "0,x,2", limits.ENV_THREADS: ""}
    )
    assert applied.ram_bytes is None
    assert applied.threads is None
    assert applied.core_ids == [0, 2]


def test_nonsense_values_are_refused_not_applied():
    assert limits.limit_memory(0) == (None, "ignored memory limit 0")
    assert limits.set_thread_env(0) == (None, "ignored thread count 0")
    assert limits.set_affinity([]) == (None, None)


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is only enforced on Linux")
def test_memory_cap_actually_kills_an_over_budget_allocation():
    # 256 MiB of budget, then ask for 1 GiB. If the cap were merely advertised
    # and not enforced, this would succeed and print "allocated".
    program = (
        "from node.core import limits;"
        "limits.apply_from_env();"
        "import numpy;"
        "a = numpy.ones(1024**3 // 8);"
        "print('allocated', a.sum())"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", limits.ENV_RAM_BYTES: str(256 * 1024**2)},
    )
    assert "allocated" not in result.stdout
    assert result.returncode != 0
    # Assert *why* it failed. Without this the test would pass just as happily
    # on an ImportError, which proves nothing about the cap.
    assert "Unable to allocate" in result.stderr or "MemoryError" in result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is only enforced on Linux")
def test_within_budget_allocation_still_succeeds():
    # The mirror image, and the more important of the two: a cap that stops
    # legitimate work is just a broken node.
    program = (
        "from node.core import limits;"
        "limits.apply_from_env();"
        "import numpy;"
        "a = numpy.ones(8 * 1024**2 // 8);"
        "print('allocated', int(a.sum()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", limits.ENV_RAM_BYTES: str(2 * 1024**3)},
    )
    assert result.returncode == 0, result.stderr
    assert "allocated 1048576" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows-only")
def test_job_object_limit_reads_back():
    # The struct layout is the thing most likely to be wrong, and a wrong
    # layout fails silently -- SetInformationJobObject returns success having
    # written the number into the wrong field. _limit_memory_windows verifies
    # by reading back, so a mismatch surfaces as a failure string here.
    note, problem = limits.limit_memory(512 * 1024**2)
    assert problem is None, problem
    assert "ProcessMemoryLimit" in note


@pytest.mark.skipif(not hasattr(__import__("os"), "sched_setaffinity"), reason="no affinity API")
def test_affinity_restricts_the_visible_cores():
    import os

    original = os.sched_getaffinity(0)
    target = sorted(original)[:1]
    try:
        note, problem = limits.set_affinity(target)
        assert problem is None, problem
        assert note == f"affinity={target[0]}"
        assert os.sched_getaffinity(0) == set(target)
    finally:
        os.sched_setaffinity(0, original)
