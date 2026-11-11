"""Equivalent standalone example using the next-generation architecture."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neuromsi import AlaisBurr2004, Audio, Backend, Visual


ears = Audio(position=-5, intensity=1, sigma=3)
eyes = Visual(position=5, intensity=1, sigma=3)
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

ax = plt.subplot()
result.plot.linep(ax=ax)
ax.set_ylabel("predicted probability")
ax.set_xlabel("stimulus location (deg)")
ax.set_title("Next-generation AlaisBurr2004")
plt.show()
