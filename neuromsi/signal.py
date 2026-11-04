"""Signals: what a stimulus becomes once it is encoded for a model.

A signal is the contract between the stimuli and the integrators. A
:class:`~neuromsi.stimulus.Stimulus` describes an event of the world and
encodes it as a signal; an integrator consumes signals without knowing
anything about the stimuli that produced them.

There is one kind of signal per kind of model, and the Backend picks the
right one according to the ``model_type`` of the integrator:

- :class:`NeuralSignal` for neural models (time-by-neuron input).
- :class:`BayesianSignal` for Bayesian models (variance and measurement).

Both are immutable and validate their content when they are created.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeuralSignal:
    """Encoded external sensory drive for one modality.

    ``values`` is a time-by-neuron array. Network state and synaptic
    connectivity belong to the integrator and are deliberately absent.

    Attributes
    ----------
    modality : str
        Name of the sensory modality (e.g. "visual"). The integrator uses
        it to route the signal to the matching layer.
    values : numpy.ndarray
        The input to each neuron at each time step, with shape
        ``(time_steps, neurons)``.

    Raises
    ------
    ValueError
        If ``values`` is not a 2D array or ``modality`` is empty.
    """

    modality: str
    values: np.ndarray

    def __post_init__(self):
        # Accept any array-like (e.g. a list of lists), but keep an array.
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError(
                "NeuralSignal.values must be a 2D array with shape "
                "(time_steps, neurons)."
            )
        if not self.modality:
            raise ValueError("NeuralSignal.modality cannot be empty.")
        # The dataclass is frozen, so the converted array has to be stored
        # bypassing the blocked ``__setattr__``.
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class BayesianSignal:
    """Sensory measurement of one modality for a Bayesian model.

    ``values`` is a mapping with the two pieces a causal-inference model
    needs: the stimulus uncertainty (``"var"``, the variance of the
    stimulus) and the unisensory measurement drawn from it (``"input"``).
    There is no time-by-neuron matrix here: the Bayesian model works on
    simulations, not on a neural grid.

    Attributes
    ----------
    modality : str
        Name of the sensory modality (e.g. "auditory"). The integrator
        uses it to tell the two measurements apart.
    values : dict
        Must contain:

        - ``"var"``: positive scalar. Variance of the stimulus, i.e. the
          square of its ``sigma``.
        - ``"input"``: float or numpy.ndarray. The unisensory measurement,
          one value per simulation (a scalar if the model runs without
          noise).

    Raises
    ------
    ValueError
        If ``modality`` is empty, if ``values`` lacks ``"var"`` or
        ``"input"``, or if ``"var"`` is not a positive scalar.
    """

    modality: str
    values: dict

    def __post_init__(self):
        if not self.modality:
            raise ValueError("BayesianSignal.modality cannot be empty.")
        # Work on a copy so the caller cannot change the signal afterwards
        # by modifying the dict it passed in.
        values = dict(self.values)
        missing = {"var", "input"}.difference(values)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                "BayesianSignal.values must contain the "
                f"entries {missing_str!r}."
            )
        # A variance of zero (or less) would break the precision
        # (1 / variance) computations of the model.
        if np.ndim(values["var"]) != 0 or values["var"] <= 0:
            raise ValueError(
                "BayesianSignal.values['var'] must be a positive scalar. "
                f"Got {values['var']!r}."
            )
        object.__setattr__(self, "values", values)
