from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stimulus import Stimulus


@dataclass
class Signal:
    """
    Packages ONE stimulus (payload) together with the information needed to
    integrate it (metadata): the unimodal temporal matrix and the synapses
    of its layer (lateral, cross-modal, feedforward and the multisensory
    layer's ones). Stimulus.generate_signal() builds one per stimulus and
    Backend passes them to integrate().

    The unimodal_matrix, latsynapses, etc. properties are read shortcuts
    over metadata.
    """

    payload: Stimulus
    metadata: dict

    @property
    def unimodal_matrix(self):
        return self.metadata["unimodal_matrix"]

    @property
    def latsynapses(self):
        return self.metadata["latsynapses"]

    @property
    def crossmodal_synapses(self):
        return self.metadata["crossmodal_synapses"]

    @property
    def feedforward_synapses(self):
        return self.metadata["feedforward_synapses"]

    @property
    def multi_latsynapses(self):
        return self.metadata["multi_latsynapses"]