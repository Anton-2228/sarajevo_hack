"""Trains in a process of its own. Not meant to be imported by callers.

Invoked as ``python -m node.core._worker <spec> <result>``, exchanging pickled
files with the parent.

Why a subprocess rather than ``multiprocessing``: the "spawn" start method
re-imports the parent's ``__main__`` in the child, which breaks outright when
``__main__`` is not a file (a REPL, a notebook, a here-document) and, worse,
spawns processes without end when a script calls train() outside an
``if __name__ == "__main__"`` guard. A plain subprocess has neither problem, so
the core stays safe to embed anywhere.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path


def main(argv: list[str]) -> int:
    # Before anything imports numpy or fastText: OpenMP reads its thread count
    # when the runtime loads, and an address-space cap installed after numpy has
    # reserved its arenas trips on allocations that were never the problem.
    from node.core import limits

    applied = limits.apply_from_env()

    spec_path, result_path = Path(argv[0]), Path(argv[1])
    samples, config, out_dir = pickle.loads(spec_path.read_bytes())

    try:
        from node.core.classifier import _fit

        result = _fit(samples, config, out_dir)
        payload: tuple[str, object] = ("ok", replace(result, limits=applied))
    except Exception as error:  # noqa: BLE001 - forwarded to the parent verbatim
        try:
            pickle.dumps(error)
            payload = ("error", error)
        except Exception:  # noqa: BLE001 - some exceptions do not survive pickling
            payload = ("error", RuntimeError(f"{type(error).__name__}: {error}"))

    result_path.write_bytes(pickle.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
