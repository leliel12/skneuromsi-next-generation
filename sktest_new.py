
import matplotlib.pyplot as plt
import numpy as np

from neuromsi import Audio, Cuppini2017, Visual, Backend

## Codigo de testeo de prototipo actual para comparar resultados con los de la libreria original
cerebro = Cuppini2017(
    neurons=90,
)

oreja = Audio(position=35, intensity=28, sigma=32)
ojos = Visual(position=52, intensity=27, sigma=4)

model = Backend(stimuli=[ojos, oreja], 
                integrator=cerebro, 
                position_range=(0, 90),
                )

res = model.run()
# Results plot
ax1 = plt.subplot()
res.plot.linep(ax=ax1)
ax1.set_ylabel("neural activity")
ax1.set_xlabel("stimulus location (deg)")

plt.show()