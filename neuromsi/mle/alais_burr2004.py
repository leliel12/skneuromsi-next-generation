#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

"""Maximum-likelihood estimator model of Alais and Burr (2004)."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..integrator import Integrator

# =============================================================================
# CLASS
# =============================================================================


class AlaisBurr2004(Integrator):
    r"""Optimal bimodal estimator model of Alais and Burr (2004).

    This model estimates multisensory integration by combining
    unisensory estimates with weights derived from the standard
    deviations of the sensory modalities. It follows the Near-optimal
    Bimodal Integrator model described in Alais and Burr (2004) for
    integrating auditory and visual information.

    The model is decoupled from the stimuli: the Backend encodes each
    stimulus as a :class:`~neuromsi.signal.UnisensorySignal` (the
    unisensory estimate plus the stimulus variance) and passes those
    signals to :meth:`execute`. Everything that belongs to the
    integration rule (the weights, the fused variance and the estimators)
    lives here. The position and time grids, the seed and the random
    generator belong to the Backend.

    References
    ----------
    :cite:p:`alais2004ventriloquist`

    Notes
    -----
    The Near-optimal Bimodal Integrator for visual (V) and auditory (A)
    signals can be computed as:

    .. math::
        \hat{S}_{VA} = w_{V} \hat{S}_{V} + w_{A} \hat{S}_{A}

    where :math:`\hat{S}_{V}` and :math:`\hat{S}_{A}` are the unisensory
    estimates and :math:`\hat{S}_{VA}` is the multisensory estimate.

    The relative weights are:

    .. math::
        w_{A} = \frac{\sigma_{V}^{2}}{\sigma_{A}^{2} + \sigma_{V}^{2}}
        \qquad
        w_{V} = \frac{\sigma_{A}^{2}}{\sigma_{V}^{2} + \sigma_{A}^{2}}

    where :math:`\sigma_{A}` and :math:`\sigma_{V}` are the standard
    deviations of each unisensory stimulus. The variance of the fused
    estimate is:

    .. math::
        \sigma_{VA}^{2} =
        \frac{\sigma_{A}^{2} \sigma_{V}^{2}}
             {\sigma_{A}^{2} + \sigma_{V}^{2}}

    The model assumes that both signals come from a common cause, so it
    always fuses them: there is no causal-inference stage. The response
    is the probability density of each estimate sampled on the position
    grid of the Backend.
    """

    #: This model consumes unisensory signals (estimate + variance), not
    #: a time-by-neuron matrix.
    model_type = "MLE"

    def __init__(self, *, mode0="auditory", mode1="visual", noise=False):
        """
        Initializes the Alais-Burr 2004 model.

        Parameters
        ----------
        mode0 : str
            The name for the first sensory modality (e.g., "auditory").
        mode1 : str
            The name for the second sensory modality (e.g., "visual").
        noise : bool
            Whether the unisensory estimates carry noise. The Backend
            reads this value when it encodes the stimuli. With False the
            signal carries the true position of each stimulus; with True
            it carries a noisy measurement drawn around it.

        Raises
        ------
        ValueError
            If ``mode0`` and ``mode1`` name the same modality.
        """
        if mode0 == mode1:
            raise ValueError("mode0 and mode1 must name different modalities.")

        self.mode0 = mode0
        self.mode1 = mode1
        self.noise = noise

    # PROPERTIES ============================================================

    @property
    def model_parameters(self):
        """dict: Every parameter that defines this model.

        Reported by the Backend in the run parameters of the result, so
        a simulation can be reproduced from its NDResult.
        """
        return {
            "mode0": self.mode0,
            "mode1": self.mode1,
            "noise": self.noise,
        }

    # MODEL METHODS =========================================================

    def check_stimuli_compatible(self, modalities):
        """Require exactly one signal for each configured sensory modality.

        Parameters
        ----------
        modalities : iterable of str
            Modality names of the incoming signals.

        Raises
        ------
        ValueError
            If there are not exactly two signals, or if they are not one
            of ``mode0`` and one of ``mode1``.
        """
        modalities = list(modalities)
        if len(modalities) != 2:
            raise ValueError(
                "AlaisBurr2004 expects exactly 2 stimuli (one for each "
                f"sensory modality), but {len(modalities)} were received."
            )
        if set(modalities) != {self.mode0, self.mode1}:
            raise ValueError(
                f"AlaisBurr2004 expects one {self.mode0!r} and one "
                f"{self.mode1!r} signal. Received: {modalities}."
            )

    def _evaluate_gaussian(self, locations, mean, sigma):
        """Evaluates a Gaussian density on the position grid.

        Parameters
        ----------
        locations : numpy.ndarray
            The grid of positions where the density is evaluated.
        mean : float or numpy.ndarray
            The center of the Gaussian. It is a scalar for a single
            estimate, or an array with one value per simulated observer.
        sigma : float
            The standard deviation of the Gaussian.

        Returns
        -------
        numpy.ndarray
            The density at each position. If ``mean`` is an array, the
            densities of the simulated observers are averaged (a mixture
            of Gaussians).
        """
        locations = np.asarray(locations)
        mean = np.asarray(mean)
        sigma2 = np.square(sigma)
        normalization = 1 / np.sqrt(2 * np.pi * sigma2)

        # A single estimate: evaluate the Gaussian directly.
        if mean.ndim == 0:
            return normalization * np.exp(
                -(np.square(locations - mean)) / (2 * sigma2)
            )

        # One estimate per simulation: average their densities.
        distances = np.square(locations[:, None] - mean[None, :])
        densities = normalization * np.exp(-distances / (2 * sigma2))
        return np.mean(densities, axis=1)

    def unisensory_estimator(
        self, unisensory_sigma, unisensory_position, possible_locations
    ):
        """Estimates the unisensory probability density function.

        Parameters
        ----------
        unisensory_sigma : float
            The standard deviation of the sensory modality.
        unisensory_position : float or numpy.ndarray
            The position estimate of the sensory modality.
        possible_locations : numpy.ndarray
            The array of possible positions to evaluate.

        Returns
        -------
        numpy.ndarray
            The estimated probability density function for the given
            sensory modality.
        """
        return self._evaluate_gaussian(
            possible_locations, unisensory_position, unisensory_sigma
        )

    def weight_calculator(self, target_sigma, reference_sigma):
        """Calculates the weight of a sensory modality.

        Parameters
        ----------
        target_sigma : float
            The standard deviation of the target sensory modality.
        reference_sigma : float
            The standard deviation of the reference sensory modality.

        Returns
        -------
        float
            The weight of the target sensory modality.
        """
        target_var = np.square(target_sigma)
        reference_var = np.square(reference_sigma)
        return reference_var / (target_var + reference_var)

    def multisensory_estimator(
        self,
        unisensory_position_a,
        unisensory_position_b,
        unisensory_weight_a,
        unisensory_weight_b,
        multisensory_sigma,
        possible_locations,
    ):
        """Estimates the multisensory probability density function.

        Parameters
        ----------
        unisensory_position_a : float or numpy.ndarray
            The position estimate of the first sensory modality.
        unisensory_position_b : float or numpy.ndarray
            The position estimate of the second sensory modality.
        unisensory_weight_a : float
            The weight of the first sensory modality.
        unisensory_weight_b : float
            The weight of the second sensory modality.
        multisensory_sigma : float
            The standard deviation of the multisensory estimate.
        possible_locations : numpy.ndarray
            The array of possible positions to evaluate.

        Returns
        -------
        numpy.ndarray
            The estimated multisensory probability density function.
        """
        location = (
            unisensory_weight_a * unisensory_position_a
            + unisensory_weight_b * unisensory_position_b
        )
        return self._evaluate_gaussian(
            possible_locations, location, multisensory_sigma
        )

    def integrate(
        self,
        signals,
        *,
        time_range,
        time_res,
        position_range,
        position_res,
        random,
    ):
        """Runs the optimal integration over a pair of unisensory signals.

        Parameters
        ----------
        signals : dict
            Unisensory signals keyed by modality. Each one carries the
            stimulus variance (``values["var"]``) and the unisensory
            estimate (``values["input"]``) already computed by the
            stimulus.
        time_range : (float, float)
            The time grid of the simulation, given by the Backend. The
            model is static, so the distributions are written in the
            first time step.
        time_res : float
            Temporal resolution of the simulation.
        position_range : (float, float)
            The position grid of the simulation, given by the Backend.
            It is the grid where the densities are evaluated.
        position_res : float
            Spatial resolution of the simulation.
        random : numpy.random.Generator
            The random number generator of the Backend. Not used by this
            model, which is deterministic given the signals; it is
            accepted for interface compatibility.

        Returns
        -------
        tuple
            A tuple containing:

            - dict: Response dictionary with the two unisensory
              densities and the multisensory one.
            - dict: Extra information with the weight of each modality.
        """
        mode0_signal = signals[self.mode0]
        mode1_signal = signals[self.mode1]

        # The signal carries the unisensory estimate and its variance.
        mode0_input = mode0_signal.values["input"]
        mode1_input = mode1_signal.values["input"]
        mode0_var = mode0_signal.values["var"]
        mode1_var = mode1_signal.values["var"]

        mode0_sigma = np.sqrt(mode0_var)
        mode1_sigma = np.sqrt(mode1_var)

        # Grid where the probability densities are evaluated.
        possible_locations = np.arange(
            position_range[0],
            position_range[1],
            position_res,
        )

        # Unisensory densities.
        mode0_estimate = self.unisensory_estimator(
            mode0_sigma, mode0_input, possible_locations
        )
        mode1_estimate = self.unisensory_estimator(
            mode1_sigma, mode1_input, possible_locations
        )

        # Weights and fused estimate.
        mode0_weight = self.weight_calculator(mode0_sigma, mode1_sigma)
        mode1_weight = self.weight_calculator(mode1_sigma, mode0_sigma)

        multisensory_sigma = np.sqrt(
            (mode0_var * mode1_var) / (mode0_var + mode1_var)
        )

        multisensory_estimate = self.multisensory_estimator(
            mode0_input,
            mode1_input,
            mode0_weight,
            mode1_weight,
            multisensory_sigma,
            possible_locations,
        )

        # The result must be a (time, position) matrix. The model has no
        # dynamics, so the densities fill the first time step and the rest
        # stays at zero (the same convention as Kording2007).
        n_times = int(abs(time_range[1] - time_range[0]) / time_res) or 1
        n_positions = possible_locations.size

        mode0_res = np.zeros((n_times, n_positions))
        mode1_res = np.zeros((n_times, n_positions))
        multi_res = np.zeros((n_times, n_positions))

        mode0_res[0, :] = mode0_estimate
        mode1_res[0, :] = mode1_estimate
        multi_res[0, :] = multisensory_estimate

        response = {
            self.mode0: mode0_res,
            self.mode1: mode1_res,
            "multi": multi_res,
        }
        extra = {
            f"{self.mode0}_weight": mode0_weight,
            f"{self.mode1}_weight": mode1_weight,
        }

        return response, extra

    def calculate_causes(self, causes_kind, **kwargs):
        """Infers the number of causes from the multisensory estimate.

        Alais and Burr's model has no causal-inference stage: it always
        assumes that both signals come from a common cause and fuses
        them. The readout is therefore constant (one cause, with
        probability 1).

        Parameters
        ----------
        causes_kind : str
            Readout method: ``"count"`` (number of causes) or ``"prob"``
            (probability of a single cause).
        **kwargs
            The remaining run extras reported by the Backend (ignored).

        Returns
        -------
        int or float or None
            ``1`` for ``"count"``, ``1.0`` for ``"prob"``, or ``None``
            if ``causes_kind`` is not valid.
        """
        if causes_kind == "count":
            return 1
        if causes_kind == "prob":
            return 1.0
        return None
