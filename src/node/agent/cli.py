"""`node-agent` -- the console entry point.

Deliberately thin. Everything the agent does lives in `loop`, `runner` and
`api`, so that a GUI can drive the same machinery without importing any of the
argument parsing.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
from pathlib import Path

from node.agent import __version__, config as config_mod, resources
from node.agent.config import (
    DEFAULT_SERVER,
    LOG_LEVELS,
    MODES,
    SCORE_SCALES,
    UNKNOWN_METRIC_POLICIES,
    AgentConfig,
)
from node.agent.state import AgentState

LOG = logging.getLogger("node.agent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="node-agent",
        description="Turn this machine into a node of the active-learning mesh.",
    )
    parser.add_argument("--version", action="version", version=f"node-agent {__version__}")
    _add_global_flags(parser)

    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser(
        "run", help="enrol, then take rounds off the server until stopped"
    )
    _add_global_flags(run)
    _add_run_flags(run)

    info = subparsers.add_parser(
        "info", help="show detected hardware, budget and local state; no network"
    )
    _add_global_flags(info)
    _add_run_flags(info)
    info.add_argument("--json", action="store_true", help="machine-readable output")

    selftest = subparsers.add_parser(
        "selftest",
        help="run one complete round against the live server, start to finish",
    )
    _add_global_flags(selftest)
    _add_run_flags(selftest)
    selftest.add_argument("--round-id", help="name for the test round (default: random)")

    reset = subparsers.add_parser("reset", help="delete local state")
    _add_global_flags(reset)
    _add_run_flags(reset)
    reset.add_argument("--identity", action="store_true", help="forget node id and secret")
    reset.add_argument("--models", action="store_true", help="delete trained models")
    reset.add_argument("--journal", action="store_true", help="clear the task journal")

    return parser


def _add_global_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", help=f"control plane base URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--log-level", choices=LOG_LEVELS, help="default: info")
    parser.add_argument("--log-file", help="also log to this file, rotating at 5 MB")


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    identity = parser.add_argument_group("identity")
    identity.add_argument("--name", help="human label; the server derives node_id from it")
    identity.add_argument("--state-dir", help="identity, models, journal and outbox")
    identity.add_argument(
        "--reset-identity", action="store_true", help="enrol as a brand new node"
    )

    budget = parser.add_argument_group("resources")
    budget.add_argument(
        "--cores", type=int, help="CPU cores to use (default: all but one)"
    )
    budget.add_argument(
        "--ram", type=float, help="RAM ceiling in GiB (default: half the machine, max 4)"
    )
    budget.add_argument(
        "--no-enforce-limits",
        action="store_true",
        help="skip the OS-level memory cap; the budget still shapes the model",
    )

    # No dataset flags: every corpus is a server-held shard the round names.
    work = parser.add_argument_group("work")
    work.add_argument(
        "--mode",
        choices=MODES,
        help="default: auto, which obeys task.operation.train (fresh/continue/skip); "
        "train and score force one policy for debugging",
    )
    work.add_argument("--model-kinds", help="advertised at handshake (default: classifier)")
    work.add_argument("--score-scale", choices=SCORE_SCALES, help="default: raw")
    work.add_argument(
        "--unknown-metric",
        choices=UNKNOWN_METRIC_POLICIES,
        help="what to send for a required metric this node cannot compute (default: null)",
    )

    net = parser.add_argument_group("transport")
    net.add_argument("--poll-interval", type=float, help="seconds between /tasks polls")
    net.add_argument("--heartbeat-interval", type=float, help="override the server's value")
    net.add_argument("--connect-timeout", type=float)
    net.add_argument("--read-timeout", type=float)
    net.add_argument("--max-retries", type=int)

    control = parser.add_argument_group("control")
    control.add_argument("--once", action="store_true", help="handle one round, then exit")
    control.add_argument("--max-tasks", type=int, help="exit after N successful submits")
    control.add_argument(
        "--dry-run",
        action="store_true",
        help="run the full pipeline but never ack or submit",
    )
    control.add_argument("--shutdown-grace", type=float, help="seconds to finish on SIGINT")


def configure_logging(level: str, log_file: Path | None) -> None:
    root = logging.getLogger("node")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # The thread name matters: the heartbeat and the main loop interleave, and
    # untangling their lines without it is unpleasant.
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # stderr, so stdout stays clean for --json. ASCII only, because a Windows
    # cp1252 console raises UnicodeEncodeError inside the logging call itself,
    # which is an absurd way to lose a node.
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5 * 1024**2, backupCount=3, encoding="utf-8"
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # No subcommand means `run`, which is what a node operator wants by default.
    command = args.command or "run"
    if args.command is None:
        args = parser.parse_args([*(argv if argv is not None else sys.argv[1:]), "run"])
        command = "run"

    cfg = config_mod.from_args(args)
    configure_logging(cfg.log_level, cfg.log_file)

    problems = cfg.validate()
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2

    if command == "info":
        return _run_info(cfg, json_output=getattr(args, "json", False))
    if command == "reset":
        return _run_reset(cfg, args)

    # Imported here so that `info` and `reset` work even if the network stack
    # is unhappy, and so `--help` stays fast.
    from node.agent.loop import run_agent, run_selftest

    if command == "run":
        return run_agent(cfg)
    if command == "selftest":
        return run_selftest(cfg, round_id=getattr(args, "round_id", None))

    parser.error(f"unknown command {command!r}")
    return 2


def describe(cfg: AgentConfig) -> dict:
    """Everything `info` reports, as data. Shared with the startup banner."""
    hw = resources.detect_hardware(cfg.state_dir.parent)
    budget, warnings = resources.resolve_budget(cfg.cores, cfg.ram_gb, hw)
    train_config, _, _ = resources.train_config_for(budget, {})

    estimated_mb = train_config.max_bucket * train_config.dim * 4 / 1024**2

    return {
        "agent_version": __version__,
        "server_url": cfg.server_url,
        "name": cfg.name,
        "hardware": {
            "platform": hw.platform,
            "python": hw.python,
            "logical_cores": hw.logical_cores,
            "total_ram_gb": round(hw.total_ram_gb, 2),
            "disk_free_gb": hw.disk_free_gb,
            "gpus": [],
        },
        "budget": {
            "cores": budget.cores,
            "ram_gb": round(budget.ram_gb, 2),
            "core_ids": budget.core_ids,
            "train_ram_gb": round(budget.train_ram_bytes() / 1024**3, 2),
            "enforce_limits": cfg.enforce_limits,
            "warnings": warnings,
        },
        "train_config": {
            "thread": train_config.thread,
            "dim": train_config.dim,
            "max_bucket": train_config.max_bucket,
            # A ceiling, not a forecast: the actual table is sized against the
            # corpus and only clamped by this. A small dataset lands far below.
            "model_ceiling_mb": round(estimated_mb, 1),
            "isolate_training": train_config.isolate_training,
        },
        # Contract 0.7.0: there is nothing local to list. Every corpus is a
        # server-held shard the node reads per round.
        "workloads": {
            "source": "server-held shards",
            "chunks_url": f"{cfg.server_url}/shards/{{shard_id}}/chunks",
            "labels_url": f"{cfg.server_url}/shards/{{shard_id}}/labels",
        },
        "state": AgentState(cfg.state_dir).describe(),
    }


def _run_info(cfg: AgentConfig, *, json_output: bool) -> int:
    report = describe(cfg)
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    hw, budget, train = report["hardware"], report["budget"], report["train_config"]
    print(f"node-agent {report['agent_version']}  ->  {report['server_url']}")
    print(f"  name           {report['name']}")
    print(f"  platform       {hw['platform']} python {hw['python']}")
    print(
        f"  machine        {hw['logical_cores']} cores, "
        f"{hw['total_ram_gb']} GiB RAM, {hw['disk_free_gb']} GiB disk free"
    )
    print(
        f"  budget         {budget['cores']} cores, {budget['ram_gb']} GiB "
        f"({budget['train_ram_gb']} GiB to training)"
        + ("" if budget["enforce_limits"] else "  [limits NOT enforced]")
    )
    if budget["core_ids"]:
        print(f"  pinned cores   {','.join(str(c) for c in budget['core_ids'])}")
    print(
        f"  training       thread={train['thread']} dim={train['dim']} "
        f"max_bucket={train['max_bucket']} (model capped at {train['model_ceiling_mb']} MB)"
    )
    print(f"  workloads      {report['workloads']['source']}")
    print(f"                 {report['workloads']['chunks_url']}")

    state = report["state"]
    print(f"  state dir      {state['state_dir']}")
    print(f"  node_id        {state['node_id'] or '(not enrolled yet)'}")
    if state["pending_payloads"]:
        print(f"  unsent rounds  {', '.join(state['pending_payloads'])}")

    for warning in budget["warnings"]:
        print(f"  warning: {warning}")
    return 0


def _run_reset(cfg: AgentConfig, args: argparse.Namespace) -> int:
    import shutil

    if not (args.identity or args.models or args.journal):
        print(
            "error: reset needs at least one of --identity, --models, --journal",
            file=sys.stderr,
        )
        return 2

    state = AgentState(cfg.state_dir)
    if args.identity:
        state.clear_identity()
        print(f"forgot identity in {cfg.state_dir}")
    if args.models:
        shutil.rmtree(cfg.state_dir / "models", ignore_errors=True)
        print("deleted trained models")
    if args.journal:
        (cfg.state_dir / "journal.json").unlink(missing_ok=True)
        shutil.rmtree(cfg.state_dir / "outbox", ignore_errors=True)
        print("cleared the task journal and outbox")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
