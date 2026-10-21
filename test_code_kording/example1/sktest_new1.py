"""Equivalent deterministic-input example using the new architecture."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neuromsi import Audio, Backend, Kording2007, Visual


ears = Audio(position=-20, intensity=1, sigma=5)
eyes = Visual(position=25, intensity=1, sigma=5)
model = Backend(
    stimuli=[ears, eyes],
    integrator=Kording2007(
        n_sim=20_000,
        p_common=0.3,
        prior_sigma=30,
        prior_mu=0,
        strategy="selection",
        noise=False,
        dimension="space",
    ),
    seed=42,
    position_range=(-60, 60),
    position_res=1,
    time_range=(0, 1),
    time_res=1,
    causes_kind="prob",
)
result = model.run()

ax = plt.subplot()
result.plot.linep(ax=ax)
ax.set_ylabel("predicted probability")
ax.set_xlabel("stimulus location (deg)")
ax.set_title("Next-generation Kording2007 — deterministic input")
plt.show()
