"""Neural model of multisensory integration by Cuppini et al. (2017)."""

import copy

import numpy as np

from ..integrator import Integrator
from neuromsi.utils.readout_tools import (
    calculate_spatiotemporal_causes_from_peaks,
)


class Cuppini2017(Integrator):
    r"""
    Neural integrator implementing the Cuppini et al. (2017) architecture:
    two reciprocally connected unisensory layers (auditory, visual) that
    feed a third multisensory layer ("causal inference area").

    Every parameter here is specific to THIS architecture: number of
    neurons, sigmoid dynamics, lateral connectivity (Mexican Hat), and
    the strength of the cross-modal / feedforward synapses. A different
    integrator could use completely different parameters, or a different
    mechanism, and still accept the same Stimulus objects.

    Notes
    -----
    Every layer is a ring of ``neurons`` units that encode a 1D space, so
    the distance between two neurons wraps around the ring. The activity
    ``y`` of each layer follows

    .. math::
        \tau \frac{dy_j}{dt} = -y_j + F(u_j),
        \qquad F(u) = \frac{1}{1 + e^{-s (u - \theta)}}

    where the total input :math:`u_j` of neuron :math:`j` adds up:

    * the external (stimulus) input, with optional noise;
    * the lateral input from its own layer (Mexican hat);
    * the cross-modal input from the other unisensory layer (only for the
      two unisensory layers);
    * the feedforward input from both unisensory layers (only for the
      multisensory layer).

    The multisensory layer receives no external input: it only reads the
    two unisensory layers.

    References
    ----------
    Cuppini, C., Shams, L., Magosso, E., & Ursino, M. (2017). A biologically
    inspired neurocomputational model for audiovisual integration and causal
    inference. European Journal of Neuroscience, 46(9), 2481-2498.
    """

    def __init__(
        self,
        *,
        neurons=180,
        tau=(3, 15, 1),            # (mode0, mode1, multisensory)
        s=0.3,
        theta=20,
        mode0="auditory",
        mode1="visual",
        cross_modal_weight=1.4,     # W0 of Wav / Wva
        cross_modal_sigma=5,
        feedforward_weight=18,      # W0 of Wma / Wmv
        feedforward_sigma=0.5,
        lateral_params=None,        # Mexican Hat, one per layer
        noise=False,
        noise_level=0.40,
    ):
        """
        Initializes the Cuppini 2017 network.

        Parameters
        ----------
        neurons : int, optional
            Number of neurons of each layer. It must match the number of
            positions of the Backend grid (``position_range`` divided by
            ``position_res``). Default 180.
        tau : tuple of 3 floats, optional
            Time constants (ms) of the layers, in the order
            ``(mode0, mode1, multisensory)``. Default ``(3, 15, 1)``.
        s : float, optional
            Slope of the sigmoid activation function. Default 0.3.
        theta : float, optional
            Central point (threshold) of the sigmoid. Default 20.
        mode0 : str, optional
            Name of the modality of the first unisensory layer. It must
            match the ``modality`` of one of the stimuli.
            Default "auditory".
        mode1 : str, optional
            Name of the modality of the second unisensory layer.
            Default "visual".
        cross_modal_weight : float, optional
            Highest synaptic efficacy of the cross-modal synapses between
            the two unisensory layers (Wav and Wva). Default 1.4.
        cross_modal_sigma : float, optional
            Width (in neurons) of the Gaussian of the cross-modal
            synapses. Default 5.
        feedforward_weight : float, optional
            Highest synaptic efficacy of the feedforward synapses from
            each unisensory layer to the multisensory one (Wma and Wmv).
            Default 18.
        feedforward_sigma : float, optional
            Width (in neurons) of the Gaussian of the feedforward
            synapses. Default 0.5.
        lateral_params : dict or None, optional
            Mexican hat parameters of the lateral synapses of each layer.
            It must have the keys ``"mode0"``, ``"mode1"`` and
            ``"multi"``, and each value is a dict with the keys
            ``excitation_loc``, ``inhibition_loc``, ``excitation_scale``
            and ``inhibition_scale`` (see
            :meth:`Integrator._calculate_lateral_synapses`). If None, the
            values of the original paper are used.
        noise : bool, optional
            Whether to add noise to the external input of the two
            unisensory layers. Default False.
        noise_level : float, optional
            Noise amplitude as a fraction of the maximum value of the
            stimulus of each layer. Only used if ``noise`` is True.
            Default 0.40.

        Raises
        ------
        ValueError
            If ``mode0`` and ``mode1`` are the same, or if ``tau`` does
            not have exactly 3 values.
        """
        if mode0 == mode1:
            raise ValueError("mode0 and mode1 must name different modalities.")
        if len(tau) != 3:
            raise ValueError(
                "tau must have exactly 3 values: "
                "(mode0, mode1, multisensory)"
            )

        self.neurons = neurons
        self.mode0 = mode0
        self.mode1 = mode1
        self.tau = tau
        self.s = s
        self.theta = theta

        self.cross_modal_weight = cross_modal_weight
        self.cross_modal_sigma = cross_modal_sigma
        self.feedforward_weight = feedforward_weight
        self.feedforward_sigma = feedforward_sigma

        # Default Mexican Hat values, one per layer, identical to those
        # hardcoded in the original monolithic run().
        self.lateral_params = lateral_params or {
            "mode0": dict(
                excitation_loc=5, inhibition_loc=4,
                excitation_scale=3, inhibition_scale=120,
            ),
            "mode1": dict(
                excitation_loc=5, inhibition_loc=4,
                excitation_scale=3, inhibition_scale=120,
            ),
            "multi": dict(
                excitation_loc=3, inhibition_loc=2.6,
                excitation_scale=2, inhibition_scale=10,
            ),
        }

        self.noise = noise
        self.noise_level = noise_level

    @property
    def model_parameters(self):
        """dict: Every parameter that defines this architecture.

        Reported by the Backend in the run parameters of the result, so
        a simulation can be reproduced from its NDResult.
        """
        return {
            "neurons": self.neurons,
            "mode0": self.mode0,
            "mode1": self.mode1,
            "tau": self.tau,
            "s": self.s,
            "theta": self.theta,
            "cross_modal_weight": self.cross_modal_weight,
            "cross_modal_sigma": self.cross_modal_sigma,
            "feedforward_weight": self.feedforward_weight,
            "feedforward_sigma": self.feedforward_sigma,
            "lateral_params": self.lateral_params,
            "noise": self.noise,
            "noise_level": self.noise_level,
        }

    def sigmoid(self, u):
        """Sigmoid activation function F(u), shared by the 3 layers.

        Parameters
        ----------
        u : float or numpy.ndarray
            Total input of the neurons.

        Returns
        -------
        float or numpy.ndarray
            Activation of the neurons, between 0 and 1.
        """
        return 1 / (1 + np.exp(-self.s * (u - self.theta)))

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
                "Cuppini2017 expects exactly 2 stimuli (one visual "
                f"and one auditory), but {len(modalities)} were received."
            )
        if set(modalities) != {self.mode0, self.mode1}:
            raise ValueError(
                f"Cuppini2017 expects one {self.mode0!r} and one "
                f"{self.mode1!r} signal. Received: {modalities}."
            )

    def get_connectivity(self):
        """
        Builds the synaptic connectivity of the whole network.

        All the matrices are ``(neurons, neurons)``, and an entry
        ``[j, k]`` is the efficacy of the synapse from the pre-synaptic
        neuron ``k`` to the post-synaptic neuron ``j``.

        Returns
        -------
        dict
            The synapses grouped by kind:

            - ``"lateral"``: Mexican hat synapses within each layer, keyed
              by ``mode0``, ``mode1`` and ``"multi"``.
            - ``"cross_modal"``: synapses between the unisensory layers,
              keyed by the tuple ``(source modality, target modality)``.
            - ``"feedforward"``: synapses from each unisensory layer to
              the multisensory one, keyed by the modality of the source.
        """
        # --- Lateral connectivity (Mexican Hat), within each layer ---
        mode0_latsynapses = self._calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["mode0"]
        )
        mode1_latsynapses = self._calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["mode1"]
        )
        multi_latsynapses = self._calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["multi"]
        )

        # --- Cross-modal synapses (Wav, Wva) ---
        mode0_to_mode1_synapses = self._calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.cross_modal_weight,
            sigma=self.cross_modal_sigma,
        )
        mode1_to_mode0_synapses = self._calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.cross_modal_weight,
            sigma=self.cross_modal_sigma,
        )

        # --- Feedforward synapses to the multisensory layer (Wma, Wmv) ---
        mode0_to_multi_synapses = self._calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.feedforward_weight,
            sigma=self.feedforward_sigma,
        )
        mode1_to_multi_synapses = self._calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.feedforward_weight,
            sigma=self.feedforward_sigma,
        )

        return {
            "lateral": {
                self.mode0: mode0_latsynapses,
                self.mode1: mode1_latsynapses,
                "multi": multi_latsynapses,
            },
            "cross_modal": {
                (self.mode0, self.mode1): mode0_to_mode1_synapses,
                (self.mode1, self.mode0): mode1_to_mode0_synapses,
            },
            "feedforward": {
                self.mode0: mode0_to_multi_synapses,
                self.mode1: mode1_to_multi_synapses,
            },
        }

    def integrate(
        self, signals, time_range, time_res, random, **kwargs
    ):
        """
        Simulates the dynamics of the three layers over time.

        The activity of every layer is advanced with an Euler step. The
        three layers are updated synchronously: all the inputs of a time
        step are computed from the activity of the previous step.

        Parameters
        ----------
        signals : dict
            :class:`~neuromsi.signal.NeuralSignal` keyed by modality
            (``mode0`` and ``mode1``). The ``values`` of each one must be
            a ``(n_time_steps, neurons)`` array, with ``n_time_steps``
            the number of steps of ``np.arange(*time_range, time_res)``.
        time_range : (float, float)
            Start and end (ms) of the simulation.
        time_res : float
            Integration step (ms).
        random : numpy.random.Generator
            Random number generator of the Backend. It is only used if
            ``noise`` is True.
        **kwargs
            The rest of the run parameters reported by the Backend (such
            as the position grid). They do not apply to this model and
            are ignored.

        Returns
        -------
        tuple
            ``(response, extra)``. ``response`` is a dict with the
            activity of each layer (``mode0``, ``mode1`` and ``"multi"``),
            each one a ``(n_time_steps, neurons)`` array. ``extra`` is
            an empty dict: this model reports nothing else.
        """
        # Connectivity is part of the network, so the integrator owns it.
        connectivity = self.get_connectivity()
        mode0_signal = signals[self.mode0]
        mode1_signal = signals[self.mode1]

        hist_times = np.arange(
                            time_range[0], time_range[1], time_res
                        )
        n_time_steps = hist_times.size

        # --- State containers ---
        # y_* is the current activity of each layer (one value per neuron);
        # *_res stores that activity at every time step.
        z_1d = np.zeros(self.neurons)
        mode0_y = copy.deepcopy(z_1d)
        mode1_y = copy.deepcopy(z_1d)
        multi_y = copy.deepcopy(z_1d)


        z_2d = np.zeros((n_time_steps, self.neurons))
        mode0_res = copy.deepcopy(z_2d)
        mode1_res = copy.deepcopy(z_2d)
        multi_res = copy.deepcopy(z_2d)
        del z_1d, z_2d

        tau_mode0, tau_mode1, tau_multi = self.tau

        for i in range(n_time_steps):
            # In every product below, `weights * y` scales the activity of
            # the pre-synaptic neurons (columns) by their synapses, and
            # summing over axis=1 collects, for each post-synaptic neuron
            # (rows), everything that arrives to it.

            # Cross-modal input (via Wva, Wav): each unisensory layer
            # receives the activity of the other one.
            mode0_cm_input = np.sum(
                connectivity["cross_modal"][(self.mode1, self.mode0)]
                * mode1_y, axis=1
            )
            mode1_cm_input = np.sum(
                connectivity["cross_modal"][(self.mode0, self.mode1)]
                * mode0_y, axis=1
            )

            # Feedforward input to the multisensory layer: it adds up the
            # activity of both unisensory layers.
            multi_input = (
                np.sum(connectivity["feedforward"][self.mode0] * mode0_y, axis=1)
                + np.sum(connectivity["feedforward"][self.mode1] * mode1_y, axis=1)
            )

            # External (stimulus) + cross-modal input
            mode0_input = mode0_signal.values[i] + mode0_cm_input
            mode1_input = mode1_signal.values[i] + mode1_cm_input

            # Noise, if enabled: a uniform draw in [-A, +A] for every neuron,
            # with A = noise_level * (highest value of that layer's stimulus),
            # so the noise scales with the strength of the stimulus. The
            # multisensory layer gets no external input, hence no noise.
            if self.noise:
                mode0_noise = -(
                    np.max(mode0_signal.values) * self.noise_level
                ) + (
                    2 * np.max(mode0_signal.values) * self.noise_level
                ) * random.random(self.neurons)
                mode1_noise = -(
                    np.max(mode1_signal.values) * self.noise_level
                ) + (
                    2 * np.max(mode1_signal.values) * self.noise_level
                ) * random.random(self.neurons)
                mode0_input = mode0_input + mode0_noise
                mode1_input = mode1_input + mode1_noise

            # Lateral input (Mexican Hat), within each layer: how the
            # neurons of a layer excite or inhibit each other.
            la = np.sum(connectivity["lateral"][self.mode0] * mode0_y, axis=1)
            lv = np.sum(connectivity["lateral"][self.mode1] * mode1_y, axis=1)
            lm = np.sum(connectivity["lateral"]["multi"] * multi_y, axis=1)

            # Total input per layer: u_a / u_v are the two unisensory
            # layers (external + cross-modal + lateral); u_m is the
            # multisensory one (feedforward + lateral).
            u_a = la + mode0_input
            u_v = lv + mode1_input
            u_m = lm + multi_input

            # Euler step: y_new = y + dt * (1/tau) * (-y + sigmoid(u)).
            # The activity relaxes toward sigmoid(u) at a speed set by the
            # time constant tau of each layer.
            mode0_y = mode0_y + time_res * (
                (-mode0_y + self.sigmoid(u_a)) / tau_mode0
            )
            mode1_y = mode1_y + time_res * (
                (-mode1_y + self.sigmoid(u_v)) / tau_mode1
            )
            multi_y = multi_y + time_res * (
                (-multi_y + self.sigmoid(u_m)) / tau_multi
            )

            # Record the activity of this time step.
            mode0_res[i, :] = mode0_y
            mode1_res[i, :] = mode1_y
            multi_res[i, :] = multi_y

        response = {
            self.mode0: mode0_res,
            self.mode1: mode1_res,
            "multi": multi_res,
        }
        return response, {}

    def calculate_causes(
        self,
        multi,
        causes_kind,
        causes_dim,
        causes_peak_threshold,
        causes_peak_distance,
        stim_position,
        **kwargs,
    ):
        """Infers the number of causes from the multisensory layer.

        Counts the peaks of the activity of the multisensory layer: a
        single peak means a common cause (C=1) and two peaks, separate
        causes (C=2).

        Parameters
        ----------
        multi : numpy.ndarray
            Activity of the multisensory layer, ``(n_time_steps, neurons)``.
        causes_kind : str
            Readout method: ``"count"`` (number of peaks) or ``"prob"``
            (probability of a single peak, from the peak heights).
        causes_dim : str
            Dimension to read the peaks from: ``"space"`` (the activity
            profile of the last time step) or ``"time"`` (the activity
            of one neuron across time).
        causes_peak_threshold : float
            Minimum height and prominence of a peak.
        causes_peak_distance : int or None
            Minimum distance (in samples) between two peaks.
        stim_position : sequence of float
            Position of the two stimuli. Their mean is the neuron used
            when reading the ``"time"`` dimension. It is used directly as
            a neuron index, so it assumes one neuron per position unit.
        **kwargs
            The remaining run extras reported by the Backend (ignored).

        Returns
        -------
        int or float or None
            The result of the readout, or None if ``causes_kind`` or
            ``causes_dim`` is not valid.
        """
        # Neuron where the temporal readout is done: halfway between
        # the two stimuli.
        position = int(np.mean([stim_position[0], stim_position[1]]))
        return calculate_spatiotemporal_causes_from_peaks(
            mode_spatiotemporal_activity_data=multi,
            causes_kind=causes_kind,
            causes_dim=causes_dim,
            peak_threshold=causes_peak_threshold,
            peak_distance=causes_peak_distance,
            time_point=-1,
            spatial_point=position,
        )
