"""Budgets, and the hyperparameters derived from them."""

import os

import pytest

from node.agent import resources
from node.agent.resources import Budget, HardwareInfo
from node.core.limits import ENV_CORE_IDS, ENV_RAM_BYTES, ENV_THREADS
from node.core.types import TrainConfig

HW = HardwareInfo(logical_cores=8, total_ram_gb=16.0, disk_free_gb=100.0)


def test_detect_hardware_declares_no_gpu():
    # The node bakes fastText classifiers and nothing else. Claiming a GPU
    # would invite LoRA rounds it cannot do.
    hw = resources.detect_hardware()
    assert hw.gpus == []
    assert hw.logical_cores >= 1
    assert hw.total_ram_gb > 0


def test_explicit_budget_is_respected():
    budget, warnings = resources.resolve_budget(2, 4.0, HW)
    assert budget.cores == 2
    assert budget.ram_gb == pytest.approx(4.0)
    assert warnings == []


def test_unset_budget_leaves_a_core_free():
    budget, _ = resources.resolve_budget(0, 0.0, HW)
    assert budget.cores == HW.logical_cores - 1
    assert budget.ram_gb == pytest.approx(4.0)


def test_oversubscription_warns_but_is_allowed():
    # Overcommitting your own machine is your business; it should just be said
    # out loud rather than silently clamped.
    budget, warnings = resources.resolve_budget(32, 64.0, HW)
    assert budget.cores == 32
    assert any("32 cores" in w for w in warnings)
    assert any("64.0 GiB" in w for w in warnings)


def test_training_child_gets_most_but_not_all_of_the_budget():
    # The parent holds the scored corpus and the outgoing JSON, so handing the
    # whole budget to the child would leave nothing for them.
    budget = Budget(cores=2, ram_bytes=4 * 1024**3)
    assert budget.train_ram_bytes() < budget.ram_bytes
    assert budget.train_ram_bytes() == int(4 * 1024**3 * resources.TRAIN_SHARE)


def test_bucket_ceiling_scales_with_the_budget():
    small = resources.max_bucket_for(512 * 1024**2, dim=100)
    large = resources.max_bucket_for(8 * 1024**3, dim=100)
    assert small < large
    # dim doubles -> each bucket costs twice as much -> half as many fit.
    assert resources.max_bucket_for(4 * 1024**3, dim=200) < resources.max_bucket_for(
        4 * 1024**3, dim=100
    )


def test_bucket_ceiling_stays_inside_the_core_limits():
    base = TrainConfig()
    assert resources.max_bucket_for(1024, dim=100) == base.min_bucket
    assert resources.max_bucket_for(1024**4, dim=10) == base.max_bucket


def test_bucket_ceiling_keeps_the_model_within_budget():
    # The whole point: bucket * dim * 4 bytes must fit the share we allowed.
    ram = 2 * 1024**3
    dim = 100
    buckets = resources.max_bucket_for(ram, dim)
    assert buckets * dim * 4 <= ram * resources.BUCKET_SHARE + 1


def test_round_params_map_onto_the_config():
    budget, _ = resources.resolve_budget(3, 4.0, HW)
    config, warnings, effective = resources.train_config_for(
        budget, {"proxy_lr": 0.3, "epoch": 40, "dim": 64, "word_ngrams": 3, "min_count": 2}
    )
    assert config.lr == 0.3
    assert config.epoch == 40
    assert config.dim == 64
    assert config.word_ngrams == 3
    assert config.min_count == 2
    assert warnings == []
    assert effective["proxy_lr_effective"] == 0.3


def test_a_neural_learning_rate_is_clamped_and_declared():
    # The spec's own example is proxy_lr=1e-05. For fastText that is "do not
    # train at all", so obeying it silently would produce a model that fails
    # the server's reliability gate for an invisible reason.
    budget, _ = resources.resolve_budget(1, 2.0, HW)
    config, warnings, effective = resources.train_config_for(budget, {"proxy_lr": 1e-05})
    assert config.lr == resources.MIN_USABLE_LR
    assert effective["proxy_lr_effective"] == resources.MIN_USABLE_LR
    assert any("1e-05" in w for w in warnings)


def test_absurd_learning_rate_is_clamped_from_above_too():
    budget, _ = resources.resolve_budget(1, 2.0, HW)
    config, warnings, _ = resources.train_config_for(budget, {"proxy_lr": 99.0})
    assert config.lr == resources.MAX_USABLE_LR
    assert warnings


def test_thread_count_never_reaches_fasttext_as_zero():
    # thread=0 does not raise: it divides by the thread count and kills the
    # process with SIGFPE.
    budget = Budget(cores=0, ram_bytes=2 * 1024**3)
    config, _, _ = resources.train_config_for(budget, {})
    assert config.thread >= 1


def test_config_always_isolates_training():
    # Isolation is both the NaN mitigation and the only place the memory cap
    # can be applied, so no round gets to turn it off.
    budget, _ = resources.resolve_budget(2, 2.0, HW)
    config, _, _ = resources.train_config_for(budget, {"isolate_training": 0})
    assert config.isolate_training is True


def test_unknown_params_are_ignored_not_fatal():
    budget, _ = resources.resolve_budget(2, 2.0, HW)
    config, warnings, _ = resources.train_config_for(budget, {"cutoff_q": 0.1, "whatever": 3})
    assert warnings == []
    assert config.thread == 2


def test_child_env_carries_the_whole_budget():
    budget = Budget(cores=2, ram_bytes=4 * 1024**3, core_ids=[0, 1])
    env = resources.child_env(budget)
    assert env[ENV_THREADS] == "2"
    assert env[ENV_RAM_BYTES] == str(budget.train_ram_bytes())
    assert env[ENV_CORE_IDS] == "0,1"
    assert env["OMP_NUM_THREADS"] == "2"


def test_child_env_without_enforcement_still_shapes_the_work():
    env = resources.child_env(Budget(cores=2, ram_bytes=4 * 1024**3, core_ids=[0, 1]), enforce=False)
    assert ENV_RAM_BYTES not in env
    assert ENV_CORE_IDS not in env
    assert env["OMP_NUM_THREADS"] == "2"


def test_applied_env_restores_exactly(monkeypatch):
    # A leaked OMP_NUM_THREADS would quietly constrain every later round in a
    # long-lived agent, which is a miserable bug to track down.
    monkeypatch.setenv("KEEP_ME", "original")
    monkeypatch.delenv("BRAND_NEW", raising=False)

    with resources.applied_env({"KEEP_ME": "temporary", "BRAND_NEW": "yes"}):
        assert os.environ["KEEP_ME"] == "temporary"
        assert os.environ["BRAND_NEW"] == "yes"

    assert os.environ["KEEP_ME"] == "original"
    assert "BRAND_NEW" not in os.environ


def test_applied_env_restores_after_an_exception(monkeypatch):
    monkeypatch.delenv("BRAND_NEW", raising=False)
    with pytest.raises(RuntimeError):
        with resources.applied_env({"BRAND_NEW": "yes"}):
            raise RuntimeError("boom")
    assert "BRAND_NEW" not in os.environ


def test_core_ids_start_from_what_we_are_already_allowed(monkeypatch):
    # Inside a cpuset or a container the process may only own cores 2,3,6,7.
    # Asking for "the first two cores" must mean 2 and 3, not 0 and 1.
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("no affinity API")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {2, 3, 6, 7})
    assert resources.pick_core_ids(2) == [2, 3]
    assert resources.pick_core_ids(10) == [2, 3, 6, 7]


def test_current_load_is_numeric_only():
    # The heartbeat schema accepts numbers and nothing else.
    load = resources.current_load(Budget(cores=1, ram_bytes=1024**3))
    assert load
    assert all(isinstance(v, (int, float)) for v in load.values())


def test_watchdog_reports_a_real_peak_even_for_a_brief_run():
    # A training run shorter than one sampling tick used to report a peak of
    # zero, which reads as a measurement rather than the absence of one.
    from node.agent.reporting import NullReporter

    budget = Budget(cores=1, ram_bytes=8 * 1024**3)
    with resources.MemoryWatchdog(budget, NullReporter(), interval_s=30.0) as watchdog:
        pass
    assert watchdog.peak_rss_mb is not None
    assert watchdog.peak_rss_mb > 0


def test_watchdog_peak_is_none_when_nothing_was_sampled():
    from node.agent.reporting import NullReporter

    watchdog = resources.MemoryWatchdog(
        Budget(cores=1, ram_bytes=1024**3), NullReporter()
    )
    assert watchdog.peak_rss_mb is None


def test_summarize_empty_is_empty():
    assert resources.summarize([]) == {}
    assert resources.summarize([1.0, 3.0]) == {"min": 1.0, "max": 3.0, "mean": 2.0}
