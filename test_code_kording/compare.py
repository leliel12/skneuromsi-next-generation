"""Compare original and next-generation Kording2007 outputs numerically."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = ROOT.parent / "scikit-neuromsi"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ORIGINAL_ROOT))

from neuromsi import Audio, Backend, Kording2007 as NextKording2007, Visual
from skneuromsi.bayesian import Kording2007 as OriginalKording2007


SEED = 42
N_SIM = 20_000
POSITION_RANGE = (-60, 60)
POSITION_RES = 1
TIME_RANGE = (0, 1)
TIME_RES = 1
STIMULI = {
    "auditory": {"position": -10, "sigma": 3},
    "visual": {"position": 12, "sigma": 7},
}
MODEL_PARAMS = {
    "p_common": 0.6,
    "prior_sigma": 25,
    "prior_mu": 0,
    "strategy": "averaging",
    "noise": True,
    "dimension": "space",
}


original_model = OriginalKording2007(
    n=N_SIM,
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
    **MODEL_PARAMS,
    causes_kind="count",
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
    integrator=NextKording2007(n_sim=N_SIM, **MODEL_PARAMS),
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
print(f"Mean P(common cause), original: {original_result.extra_.mean_p_common_cause:.8f}")
print(f"Mean P(common cause), new:      {next_result.extra_.mean_p_common_cause:.8f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
original_result.plot.linep(ax=axes[0])
axes[0].set_title("Original Kording2007")
axes[0].set_ylabel("predicted probability")
axes[0].set_xlabel("stimulus location (deg)")
next_result.plot.linep(ax=axes[1])
axes[1].set_title("Next-generation Kording2007")
axes[1].set_ylabel("predicted probability")
axes[1].set_xlabel("stimulus location (deg)")
fig.tight_layout()
plt.show()
