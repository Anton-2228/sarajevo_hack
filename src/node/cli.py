"""Console entry point for the node's classifier core.

Round 1 is ``train``; rounds 2 and 3 are ``score`` against the model that round
1 left on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from node.core.classifier import FastTextClassifier
from node.core.dataset import load_jsonl, parse_labeled, parse_unlabeled
from node.core.types import TrainConfig


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _run_train(args: argparse.Namespace) -> int:
    records = load_jsonl(args.golden)
    samples, unparsed = parse_labeled(records)
    if not samples:
        print(f"no usable samples in {args.golden}", file=sys.stderr)
        return 1

    config = TrainConfig(
        epoch=args.epoch,
        autotune_seconds=args.autotune_seconds,
        autotune_min_samples=args.autotune_min_samples,
        autotune_model_size=args.max_model_size,
        quantize=args.quantize,
        thread=args.threads,
        verbose=2 if args.verbose else 0,
    )

    classifier, report = FastTextClassifier.train(samples, config)
    classifier.save(args.model_dir)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"trained on {report.n_train} samples, evaluated on {report.n_eval}")
    print(f"  labels      : {', '.join(report.labels)}")
    print(f"  unparsed    : {unparsed}    empty after normalization: {report.n_skipped}")
    print(f"  autotuned   : {report.autotuned}")
    print(f"  accuracy    : {_fmt(report.accuracy)}")
    print(f"  macro F1    : {_fmt(report.macro_f1)}")
    print(f"  MAE         : {_fmt(report.mae)}      (ordinal, lower is better)")
    print(f"  QWK         : {_fmt(report.qwk)}      (ordinal, higher is better)")
    print(f"  took        : {report.duration_s}s")
    print("  per class:")
    for entry in report.per_class:
        print(
            f"    {entry.label:>4}  support={entry.support:<5} "
            f"P={entry.precision:.3f} R={entry.recall:.3f} F1={entry.f1:.3f}"
        )
    print(f"saved to {Path(args.model_dir).resolve()}")
    return 0


def _run_score(args: argparse.Namespace) -> int:
    classifier = FastTextClassifier.load(args.model_dir)
    records = load_jsonl(args.input)
    samples, unparsed = parse_unlabeled(records)
    if not samples:
        print(f"no usable samples in {args.input}", file=sys.stderr)
        return 1

    scored = classifier.score(samples)
    payload = {
        "labels": classifier.labels,
        "n_samples": len(scored),
        "n_unparsed": unparsed,
        "scores": [s.to_dict() for s in scored],
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"scored {len(scored)} samples -> {Path(args.out).resolve()}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="node-clf", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="round 1: fit on golden + teacher scores")
    train.add_argument("--golden", required=True, help="JSONL with text and label fields")
    train.add_argument("--model-dir", required=True, help="where to persist the model")
    train.add_argument(
        "--epoch", type=int, default=None, help="override the size-derived epoch count"
    )
    train.add_argument("--autotune-seconds", type=int, default=60)
    train.add_argument("--autotune-min-samples", type=int, default=2000)
    train.add_argument(
        "--max-model-size",
        default=None,
        metavar="SIZE",
        help='cap the autotuned model, e.g. "50M" (autotune ignores size otherwise)',
    )
    train.add_argument(
        "--threads", type=int, default=0, help="0 means one per core; 1 makes runs reproducible"
    )
    train.add_argument("--quantize", action="store_true", help="shrink the saved model")
    train.add_argument("--json", action="store_true", help="print the report as JSON")
    train.add_argument("--verbose", action="store_true", help="show fastText progress")
    train.set_defaults(func=_run_train)

    score = subparsers.add_parser("score", help="rounds 2-3: score with the saved model")
    score.add_argument("--model-dir", required=True)
    score.add_argument("--input", required=True, help="JSONL with a text field")
    score.add_argument("--out", help="write JSON here instead of stdout")
    score.set_defaults(func=_run_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
