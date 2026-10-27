"""Equivalent standalone example using the next-generation architecture."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neuromsi import Audio, Backend, Kording2007, Visual


ears = Audio(position=-10, intensity=1, sigma=3)
eyes = Visual(position=12, intensity=1, sigma=7)
model = Backend(
    stimuli=[ears, eyes],
    integrator=Kording2007(
        n_sim=20_000,
        p_common=0.6,
        prior_sigma=25,
        prior_mu=0,
        strategy="averaging",
        noise=True,
        dimension="space",
    ),
    seed=42,
    position_range=(-60, 60),
    position_res=1,
    time_range=(0, 1),
    time_res=1,
    causes_kind="count",
)
result = model.run()

ax = plt.subplot()
result.plot.linep(ax=ax)
ax.set_ylabel("predicted probability")
ax.set_xlabel("stimulus location (deg)")
ax.set_title("Next-generation Kording2007")
plt.show()
