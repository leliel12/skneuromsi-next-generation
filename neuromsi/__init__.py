"""Multisensory models with separate stimulus, signal, integrator and backend.

Stimuli describe external events and encode them as modality-specific
sensory drive. Signals carry that drive; integrators own network dynamics
and connectivity; Backend validates inputs and coordinates each run.
"""

from .stimulus import Audio, Stimulus, Visual
from .signal import NeuralSignal, UnisensorySignal
from .integrator import Integrator
from .neural.cuppini2017 import Cuppini2017
from .bayesian.kording2007 import Kording2007
from .mle.alais_burr2004 import AlaisBurr2004
from .backend import Backend

__all__ = [
    "Stimulus",
    "Visual",
    "Audio",
    "NeuralSignal",
    "UnisensorySignal",
    "Integrator",
    "Cuppini2017",
    "Kording2007",
    "AlaisBurr2004",
    "Backend",
]
