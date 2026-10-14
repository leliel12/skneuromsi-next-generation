from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Signal:
    """Encoded external sensory drive for one modality.

    ``values`` is a time-by-neuron array. Network state and synaptic
    connectivity belong to the integrator and are deliberately absent.
    """

    modality: str
    values: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError(
                "Signal.values must be a 2D array with shape "
                "(time_steps, neurons)."
            )
        if not self.modality:
            raise ValueError("Signal.modality cannot be empty.")
        object.__setattr__(self, "values", values)
