import random

import pytest

from node.core.types import LabeledSample, TrainConfig

MARKERS = {
    "1": "volcano basalt eruption",
    "2": "sonata orchestra violin",
    "3": "mortgage dividend portfolio",
    "4": "antibiotic diagnosis surgeon",
    "5": "compiler runtime kernel",
    "6": "harvest irrigation orchard",
    "7": "marathon striker referee",
    "8": "monsoon glacier sediment",
    "9": "parliament referendum treaty",
    "10": "telescope asteroid nebula",
}

FILLER = "the report notes that a recent review of this subject was published".split()


@pytest.fixture
def separable_samples() -> list[LabeledSample]:
    """Ten labels whose marker words make them trivially separable.

    A classifier that cannot learn this has a broken pipeline, not a hard task.
    """
    rng = random.Random(7)
    samples = []
    for label, markers in MARKERS.items():
        for index in range(14):
            padding = " ".join(rng.choice(FILLER) for _ in range(rng.randint(6, 12)))
            samples.append(
                LabeledSample(
                    sample_id=f"{label}-{index}",
                    text=f"{markers} {padding}",
                    label=label,
                )
            )
    rng.shuffle(samples)
    return samples


@pytest.fixture
def fast_config() -> TrainConfig:
    """Single-threaded so training is reproducible; fastText is otherwise
    asynchronous across threads and gives different weights each run."""
    return TrainConfig(thread=1, verbose=0, retrain_on_full=False)
