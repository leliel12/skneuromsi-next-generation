"""Multisensory models with separate stimulus, signal, integrator and backend.

Stimuli describe external events and encode them as modality-specific
sensory drive. Signals carry that drive; integrators own network dynamics
and connectivity; Backend validates inputs and coordinates each run.
"""

from .stimulus import Audio, Stimulus, Visual
from .signal import BayesianSignal, NeuralSignal
from .integrator import Integrator
from .neural.cuppini2017 import Cuppini2017
from .bayesian.kording2007 import Kording2007
from .backend import Backend

__all__ = [
    "Stimulus",
    "Visual",
    "Audio",
    "NeuralSignal",
    "BayesianSignal",
    "Integrator",
    "Cuppini2017",
    "Kording2007",
    "Backend",
]
