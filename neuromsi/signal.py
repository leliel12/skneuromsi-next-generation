from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeuralSignal:
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
                "NeuralSignal.values must be a 2D array with shape "
                "(time_steps, neurons)."
            )
        if not self.modality:
            raise ValueError("NeuralSignal.modality cannot be empty.")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class BayesianSignal:
    """Sensory measurement of one modality for a Bayesian model.

    ``values`` is a mapping with the two pieces a causal-inference model
    needs: the stimulus uncertainty (``"var"``, the variance of the
    stimulus) and the unisensory measurement drawn from it (``"input"``).
    There is no time-by-neuron matrix here: the Bayesian model works on
    simulations, not on a neural grid.
    """

    modality: str
    values: dict

    def __post_init__(self):
        if not self.modality:
            raise ValueError("BayesianSignal.modality cannot be empty.")
        values = dict(self.values)
        missing = {"var", "input"}.difference(values)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                "BayesianSignal.values must contain the "
                f"entries {missing_str!r}."
            )
        if np.ndim(values["var"]) != 0 or values["var"] <= 0:
            raise ValueError(
                "BayesianSignal.values['var'] must be a positive scalar. "
                f"Got {values['var']!r}."
            )
        object.__setattr__(self, "values", values)
