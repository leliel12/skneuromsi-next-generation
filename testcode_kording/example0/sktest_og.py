"""Standalone example using the original Kording implementation."""

import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT = ROOT.parent / "scikit-neuromsi"
sys.path.insert(0, str(ORIGINAL_ROOT))

from skneuromsi.bayesian import Kording2007


model = Kording2007(
    n=20_000,
    position_range=(-60, 60),
    position_res=1,
    time_range=(0, 1),
    time_res=1,
    seed=42,
)
result = model.run(
    auditory_position=-10,
    visual_position=12,
    auditory_sigma=3,
    visual_sigma=7,
    p_common=0.6,
    prior_sigma=25,
    prior_mu=0,
    strategy="averaging",
    noise=True,
    causes_kind="count",
    dimension="space",
)

ax = plt.subplot()
result.plot.linep(ax=ax)
ax.set_ylabel("predicted probability")
ax.set_xlabel("stimulus location (deg)")
ax.set_title("Original Kording2007")
plt.show()
