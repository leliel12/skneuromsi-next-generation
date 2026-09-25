"""
Refactor of the Cuppini et al. (2017) model into three separate pieces,
each class in its own module:

1. stimulus.py:
   The stimuli (Stimulus and its subclasses Visual, Audio) represent the
   external sensory event (position, intensity, uncertainty, timing) and
   know how to render themselves as a neural input
   (calculate_stimuli_input, create_unimodal_stimuli_matrix). They know
   nothing about the integration dynamics or the network connectivity.

2. integrator.py + cuppini2017.py:
   Integrator is the generic base class: it contains the methods to
   compute the synaptic connectivity. Cuppini2017 inherits from it and
   adds what is specific to this architecture (plus signal.py, which
   packages a stimulus together with its synapses for integration).

3. wiring.py:
   Backend connects stimuli with the integrator: it validates the
   stimuli, asks them for their Signal (via Stimulus.generate_signal) and
   integrates. It has no architecture or simulation parameters of its own.
"""

from .stimulus import Audio, Stimulus, Visual
from .signal import Signal
from .integrator import Integrator
from .neural.cuppini2017 import Cuppini2017
from .backend import Backend

__all__ = [
    "Stimulus",
    "Visual",
    "Audio",
    "Signal",
    "Integrator",
    "Cuppini2017",
    "Backend",
]