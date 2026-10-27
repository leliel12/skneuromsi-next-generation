"""Original Kording example with deterministic sensory measurements."""

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
    auditory_position=-20,
    visual_position=25,
    auditory_sigma=5,
    visual_sigma=5,
    p_common=0.3,
    prior_sigma=30,
    prior_mu=0,
    strategy="selection",
    noise=False,
    causes_kind="prob",
    dimension="space",
)

ax = plt.subplot()
result.plot.linep(ax=ax)
ax.set_ylabel("predicted probability")
ax.set_xlabel("stimulus location (deg)")
ax.set_title("Original Kording2007 — deterministic input")
plt.show()
