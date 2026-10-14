"""Run both Cuppini2017 implementations and plot the same time index."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT = ROOT.parent / "scikit-neuromsi"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ORIGINAL_ROOT))

from neuromsi import Audio, Backend, Cuppini2017 as NextCuppini2017, Visual
from skneuromsi.neural import Cuppini2017 as OriginalCuppini2017


FIXED_TIME_INDEX = 9999
TIME_RANGE = (0, 100)
TIME_RES = 0.01


# Original implementation: parameters equivalent to sktest_og1.py.
original_model = OriginalCuppini2017(
    neurons=180,
    tau=(3, 15, 1),
    s=0.3,
    theta=20,
    seed=42,
    time_range=TIME_RANGE,
    time_res=TIME_RES,
)
original_result = original_model.run(
    auditory_position=100,
    visual_position=90,
    auditory_sigma=32,
    visual_sigma=4,
    auditory_intensity=28,
    visual_intensity=27,
    auditory_duration=100,
    visual_duration=100,
    noise=True,
    noise_level=0.40,
)


# Next-generation implementation: parameters equivalent to sktest_new1.py.
ears = Audio(position=100, intensity=28, sigma=32, onset=0, duration=100)
eyes = Visual(position=90, intensity=27, sigma=4, onset=0, duration=100)
next_model = Backend(
    stimuli=[ears, eyes],
    integrator=NextCuppini2017(
        neurons=180,
        tau=(3, 15, 1),
        s=0.3,
        theta=20,
        cross_modal_weight=1.4,
        feedforward_weight=18,
        noise=True,
        noise_level=0.40,
    ),
    seed=42,
    time_range=TIME_RANGE,
    time_res=TIME_RES,
)
next_result = next_model.run()


for name, result in (
    ("Original", original_result),
    ("Next-generation", next_result),
):
    if FIXED_TIME_INDEX not in result.times_:
        raise ValueError(
            f"{name} result has no time index {FIXED_TIME_INDEX}; "
            f"available indices are {result.times_[0]} through "
            f"{result.times_[-1]}."
        )


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
original_result.plot.linep(time=FIXED_TIME_INDEX, ax=axes[0])
axes[0].set_title(f"Original — step {FIXED_TIME_INDEX}")
axes[0].set_ylabel("neural activity")
axes[0].set_xlabel("stimulus location (deg)")

next_result.plot.linep(time=FIXED_TIME_INDEX, ax=axes[1])
axes[1].set_title(f"Next-generation — step {FIXED_TIME_INDEX}")
axes[1].set_ylabel("neural activity")
axes[1].set_xlabel("stimulus location (deg)")

fig.tight_layout()
plt.show()
