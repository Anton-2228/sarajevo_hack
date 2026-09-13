"""Regenerate data/tiny_golden.jsonl, the synthetic stand-in for golden data.

The generator itself lives in `node.agent.datasets` so the agent can synthesize
a dataset on a machine that has none. This script is the checked-in entry point
for refreshing the committed file.

Deterministic: rerunning reproduces the committed file byte for byte.
"""

from __future__ import annotations

from pathlib import Path

from node.agent.datasets import synthesize_golden

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "tiny_golden.jsonl"


def main() -> None:
    path = synthesize_golden(OUTPUT)
    print(f"wrote {sum(1 for _ in path.open(encoding='utf-8'))} records to {path}")


if __name__ == "__main__":
    main()
