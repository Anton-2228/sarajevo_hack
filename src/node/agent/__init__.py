"""The node agent: turns this machine into a worker for the control plane.

`node.core` knows how to train and score. This package knows how to enrol with
the server, take rounds off it one at a time, and give the scores back without
spending more of the machine than the operator allowed.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
