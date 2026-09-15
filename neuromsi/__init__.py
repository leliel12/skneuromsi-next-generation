"""
Refactor del modelo de Cuppini et al. (2017) en tres piezas separadas,
cada clase en su propio modulo:

1. stimulus.py:
   Los estimulos (Stimulus y sus subclases Visual, Audio) representan
   UNICAMENTE el evento sensorial externo (posicion, intensidad,
   incertidumbre, timing). No saben nada de neuronas, arreglos,
   sigmoides ni sinapsis.

2. integrator.py + cuppini2017.py:
   Integrator es la clase base generica: contiene los metodos para
   calcular la conectividad sinaptica. Cuppini2017 hereda de ella y
   agrega lo especifico de esta arquitectura (mas signal.py, que
   empaqueta un estimulo con sus sinapsis para integrar).

3. wiring.py:
   Solo conecta estimulos con el integrador, enrutando cada estimulo a
   la entrada correspondiente segun su `.modality`. No tiene parametros
   propios de arquitectura ni de simulacion.
"""

from .stimulus import Audio, Stimulus, Visual
from .signal import Signal
from .integrator import Integrator
from .cuppini2017 import Cuppini2017
from .wiring import Wiring

__all__ = [
    "Stimulus",
    "Visual",
    "Audio",
    "Signal",
    "Integrator",
    "Cuppini2017",
    "Wiring",
]