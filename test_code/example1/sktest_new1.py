import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neuromsi import Audio, Cuppini2017, Visual, Backend

ears = Audio(position=100, intensity=28, sigma=32, onset=0, duration=100)
eyes = Visual(position=90, intensity=27, sigma=4, onset=0, duration=100)

# --- 2. Integrador ----------------------------------------------------
brain = Cuppini2017(
    neurons=180,
    tau=(3, 15, 1),
    s=0.3,
    theta=20,
    cross_modal_weight=1.4,
    feedforward_weight=18,
    noise=True,
    noise_level=0.40,
)

# --- 3. Wiring: conecta estimulos <-> integrador -----------------------
model = Backend(stimuli=[eyes, ears], 
                integrator=brain, 
                seed=42,
                time_range=(0, 100),
                time_res=0.01,
                )

res = model.run()
# Results plot
ax1 = plt.subplot()
res.plot.linep(ax=ax1)
ax1.set_ylabel("neural activity")
ax1.set_xlabel("stimulus location (deg)")

plt.show()
