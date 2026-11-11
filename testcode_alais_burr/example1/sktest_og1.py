"""Original Alais-Burr example with unequal sensory reliabilities."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT = ROOT.parent / "scikit-neuromsi"
sys.path.insert(0, str(ORIGINAL_ROOT))

from skneuromsi.mle import AlaisBurr2004


model = AlaisBurr2004(
    position_range=(-20, 20),
    position_res=0.1,
    time_range=(0, 1),
    time_res=1,
    seed=42,
)
result = model.run(
    auditory_position=-10,
    visual_position=10,
    auditory_sigma=8,
    visual_sigma=4,
    noise=False,
)
print(
    "weights:",
    f"{result.extra_['auditory_weight']:.2f} auditory,",
    f"{result.extra_['visual_weight']:.2f} visual",
)

ax = plt.subplot()
result.plot.linep(ax=ax)
ax.set_ylabel("predicted probability")
ax.set_xlabel("stimulus location (deg)")
ax.set_title("Original AlaisBurr2004 — unequal sigmas")
plt.show()
