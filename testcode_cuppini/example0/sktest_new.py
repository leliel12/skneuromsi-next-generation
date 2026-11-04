
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neuromsi import Audio, Cuppini2017, Visual, Backend

brain = Cuppini2017(
    neurons=90,
)

ears = Audio(position=35, intensity=28, sigma=32)
eyes = Visual(position=52, intensity=27, sigma=4)

model = Backend(stimuli=[ears, eyes], 
                integrator=brain, 
                position_range=(0, 90),
                )


res = model.run()
# Results plot
ax1 = plt.subplot()
res.plot.linep(ax=ax1)
ax1.set_ylabel("neural activity")
ax1.set_xlabel("stimulus location (deg)")

plt.show()
