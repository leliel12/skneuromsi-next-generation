"""Next-generation AlaisBurr2004 with unequal sensory reliabilities."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neuromsi import AlaisBurr2004, Audio, Backend, Visual


# Audition is less reliable than vision (sigma 8 vs 4), so the fused
# estimate should be pulled toward the visual position.
ears = Audio(position=-10, intensity=1, sigma=8)
eyes = Visual(position=10, intensity=1, sigma=4)
model = Backend(
    stimuli=[ears, eyes],
    integrator=AlaisBurr2004(noise=False),
    seed=42,
    position_range=(-20, 20),
    position_res=0.1,
    time_range=(0, 1),
    time_res=1,
    causes_kind="count",
)
result = model.run()
print(
    "weights:",
    f"{result.extra_['auditory_weight']:.2f} auditory,",
    f"{result.extra_['visual_weight']:.2f} visual",
)

ax = plt.subplot()
result.plot.linep(ax=ax)
ax.set_ylabel("predicted probability")
ax.set_xlabel("stimulus location (deg)")
ax.set_title("Next-generation AlaisBurr2004 — unequal sigmas")
plt.show()
