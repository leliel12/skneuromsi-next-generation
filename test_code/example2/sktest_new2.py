import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neuromsi import Audio, Cuppini2017, Visual, Backend

oreja = Audio(position=20, intensity=28, sigma=32, onset=0, duration=100)
ojos = Visual(position=65, intensity=27, sigma=4, onset=0, duration=100)

cerebro = Cuppini2017(
    neurons=90,
)

model = Backend(stimuli=[ojos, oreja], 
                integrator=cerebro, 
                causes_kind="count",
                causes_dim="space",
                causes_peak_threshold=0.15,
                causes_peak_distance=10,
                position_range=(0,90)
                )
        
res = model.run()
# Results plot
ax1 = plt.subplot()
res.plot.linep(ax=ax1)
ax1.set_ylabel("neural activity")
ax1.set_xlabel("stimulus location (deg)")

plt.show()
