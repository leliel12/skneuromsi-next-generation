#from skneuromsi.neural import Cuppini2017
import matplotlib.pyplot as plt
import numpy as np

from neuromsi import Audio, Cuppini2017, Visual, Backend


from neuromsi import *
"""
# Model setup
model_cuppini2017 = Cuppini2017(neurons=90, 
                                position_range=(0, 90))

# Model execution
res = model_cuppini2017.run(auditory_position=35, 
                            visual_position=52)

# Results plot
ax1 = plt.subplot()
res.plot.linep(ax=ax1)
ax1.set_ylabel("neural activity")
ax1.set_xlabel("stimulus location (deg)") """

"""
Test rapido del pipeline Stimulus -> Wiring -> Cuppini2017.

A diferencia de la version vieja (que hacia `from prot import *`, el
modulo monolitico original), esta usa el paquete `neuromsi` ya separado
en piezas: stimulus.py, signal.py, integrator.py, cuppini2017.py y
wiring.py.
"""


# --- 1. Estimulos ---------------------------------------------------------
# El primer estimulo de la lista va a "mode_1" y el segundo a "mode_2"
# dentro del integrador; Wiring despues los vuelve a etiquetar con su
# `.modality` real ("auditory" / "visual") en la respuesta final.
oreja = Audio(position=100, intensity=28, sigma=32, onset=0, duration=100)
ojos = Visual(position=90, intensity=27, sigma=4, onset=0, duration=100)

# --- 2. Integrador ----------------------------------------------------
cerebro = Cuppini2017(
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
model = Backend(stimuli=[ojos, oreja], 
                integrator=cerebro, seed=42,
                time_range=(0, 100),
                time_res=0.01,
                )

res, extra = model.run()
print(res)
causes = model.calculate_causes(res, extra)
print(causes)