"""Compare original and next-generation AlaisBurr2004 outputs numerically."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = ROOT.parent / "scikit-neuromsi"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ORIGINAL_ROOT))

from neuromsi import AlaisBurr2004 as NextAlaisBurr2004, Audio, Backend, Visual
from skneuromsi.mle import AlaisBurr2004 as OriginalAlaisBurr2004


SEED = 42
POSITION_RANGE = (-20, 20)
POSITION_RES = 0.1
TIME_RANGE = (0, 1)
TIME_RES = 1
STIMULI = {
    "auditory": {"position": -10, "sigma": 8},
    "visual": {"position": 10, "sigma": 4},
}


original_model = OriginalAlaisBurr2004(
    position_range=POSITION_RANGE,
    position_res=POSITION_RES,
    time_range=TIME_RANGE,
    time_res=TIME_RES,
    seed=SEED,
)
original_result = original_model.run(
    auditory_position=STIMULI["auditory"]["position"],
    auditory_sigma=STIMULI["auditory"]["sigma"],
    visual_position=STIMULI["visual"]["position"],
    visual_sigma=STIMULI["visual"]["sigma"],
    noise=False,
)

next_model = Backend(
    stimuli=[
        Audio(
            position=STIMULI["auditory"]["position"],
            intensity=1,
            sigma=STIMULI["auditory"]["sigma"],
        ),
        Visual(
            position=STIMULI["visual"]["position"],
            intensity=1,
            sigma=STIMULI["visual"]["sigma"],
        ),
    ],
    integrator=NextAlaisBurr2004(noise=False),
    seed=SEED,
    position_range=POSITION_RANGE,
    position_res=POSITION_RES,
    time_range=TIME_RANGE,
    time_res=TIME_RES,
    causes_kind="count",
)
next_result = next_model.run()

mode_names = ["auditory", "visual", "multi"]
original_data = original_result.to_xarray().sel(modes=mode_names).values
next_data = next_result.to_xarray().sel(modes=mode_names).values
np.testing.assert_allclose(next_data, original_data, rtol=0, atol=0)
print("Outputs match exactly for auditory, visual, and multi modes.")
print(
    "Weights, original:",
    f"{original_result.extra_['auditory_weight']:.4f} auditory,",
    f"{original_result.extra_['visual_weight']:.4f} visual",
)
print(
    "Weights, new:     ",
    f"{next_result.extra_['auditory_weight']:.4f} auditory,",
    f"{next_result.extra_['visual_weight']:.4f} visual",
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
original_result.plot.linep(ax=axes[0])
axes[0].set_title("Original AlaisBurr2004")
axes[0].set_ylabel("predicted probability")
axes[0].set_xlabel("stimulus location (deg)")
next_result.plot.linep(ax=axes[1])
axes[1].set_title("Next-generation AlaisBurr2004")
axes[1].set_ylabel("predicted probability")
axes[1].set_xlabel("stimulus location (deg)")
fig.tight_layout()
plt.show()
