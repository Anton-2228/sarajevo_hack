"""Settings must survive a round trip, and must never stop the window opening."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from node.agent.config import ENV_PREFIX, AgentConfig
from node.agent.resources import HardwareInfo
from node.gui import settings as settings_mod
from node.gui.presets import LARGE, MEDIUM, SMALL, preset_budget
from node.gui.settings import (
    SETTINGS_VERSION,
    GuiSettings,
    build_config,
    load,
    problems_by_field,
    resolved_budget,
    save,
)

HW = HardwareInfo(logical_cores=8, total_ram_gb=16.0, disk_free_gb=50.0)


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "gui.json"


def test_round_trip(path: Path) -> None:
    original = GuiSettings(
        preset=LARGE,
        advanced=True,
        cores=3,
        ram_gb=2.5,
        autostart=True,
        tray_hint_shown=True,
        window_geometry_hex="deadbeef",
        last_run_completed=7,
        last_run_failed=1,
        last_run_finished_at=1_700_000_000.0,
    )
    save(original, path)

    assert load(path) == original


def test_a_missing_file_is_defaults(path: Path) -> None:
    assert load(path) == GuiSettings()


def test_corrupt_json_is_defaults_not_an_exception(path: Path) -> None:
    path.write_text('{"preset": "large", "cor', encoding="utf-8")

    assert load(path) == GuiSettings()


def test_an_older_version_is_defaults(path: Path) -> None:
    path.write_text(
        json.dumps({"version": SETTINGS_VERSION - 1, "preset": LARGE}), encoding="utf-8"
    )

    assert load(path) == GuiSettings()


def test_unknown_keys_are_ignored(path: Path) -> None:
    path.write_text(
        json.dumps(
            {"version": SETTINGS_VERSION, "preset": SMALL, "from_the_future": [1, 2]}
        ),
        encoding="utf-8",
    )

    assert load(path).preset == SMALL


def test_a_wrong_type_is_ignored_and_the_rest_survives(path: Path) -> None:
    path.write_text(
        json.dumps({"version": SETTINGS_VERSION, "preset": SMALL, "cores": "four"}),
        encoding="utf-8",
    )

    loaded = load(path)
    assert loaded.preset == SMALL
    assert loaded.cores == 0


def test_a_nonsense_preset_falls_back(path: Path) -> None:
    path.write_text(
        json.dumps({"version": SETTINGS_VERSION, "preset": "enormous"}), encoding="utf-8"
    )

    assert load(path).preset == MEDIUM


def test_advanced_beats_the_preset() -> None:
    settings = GuiSettings(preset=LARGE, advanced=True, cores=2, ram_gb=1.5)

    assert resolved_budget(settings, HW) == (2, 1.5)


def test_the_preset_is_used_when_advanced_is_off() -> None:
    settings = GuiSettings(preset=SMALL, advanced=False, cores=99, ram_gb=99.0)
    budget = preset_budget(SMALL, HW)

    assert resolved_budget(settings, HW) == (budget.cores, budget.ram_gb)


def test_the_default_config_validates_clean(tmp_path: Path) -> None:
    cfg = build_config(GuiSettings(), HW, state_dir=tmp_path)

    assert cfg.validate() == []


def test_auto_values_validate_clean(tmp_path: Path) -> None:
    """0 means "work it out from the machine"; the spin boxes' "auto" relies on it."""
    settings = GuiSettings(advanced=True, cores=0, ram_gb=0.0)
    cfg = build_config(settings, HW, state_dir=tmp_path)

    assert cfg.validate() == []


def test_one_shot_flags_are_never_persisted(path: Path, tmp_path: Path) -> None:
    """A reset_identity read back off disk would mint a new node every launch."""
    save(GuiSettings(), path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert "reset_identity" not in raw
    assert "once" not in raw
    assert "max_tasks" not in raw
    assert build_config(GuiSettings(), HW, state_dir=tmp_path).reset_identity is False


def test_reset_identity_is_opt_in_per_start(tmp_path: Path) -> None:
    cfg = build_config(GuiSettings(), HW, state_dir=tmp_path, reset_identity=True)

    assert cfg.reset_identity is True


def test_the_server_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_PREFIX + "SERVER", "https://example.test/")

    assert settings_mod.server_url() == "https://example.test"


def test_problems_are_routed_to_their_widget() -> None:
    cfg = AgentConfig(name="", cores=-1, ram_gb=0.1)
    by_field, other = problems_by_field(cfg)

    assert "cores" in by_field
    assert "ram_gb" in by_field
    assert "name" in by_field
    assert other == []


def test_every_validation_message_maps_to_a_field() -> None:
    """The guard against `config.py` growing a check the window cannot show.

    If this fails, `AgentConfig.validate()` gained a message whose flag is not
    in FLAG_TO_FIELD, and a user would see Start disabled with no explanation.
    """
    broken = AgentConfig(
        server_url="ftp://nope",
        name="not a valid id!",
        cores=-1,
        ram_gb=-1.0,
        mode="sideways",
        score_scale="logarithmic",
        unknown_metric="shrug",
        poll_interval_s=0.0,
        max_tasks=0,
        model_kinds=[],
    )
    problems = broken.validate()
    _, other = problems_by_field(broken)

    assert len(problems) == 10, problems
    assert other == []
