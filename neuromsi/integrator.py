"""Base class of the models that integrate multisensory signals."""

from collections.abc import Mapping

import numpy as np

from neuromsi.utils.neural_tools import calculate_neural_distance


class Integrator:
    """
    Base class of the multisensory models (neural, Bayesian and MLE).

    An integrator receives one signal per modality, already encoded by the
    stimuli, and returns the response of the model. It knows nothing about
    the stimuli themselves: it only sees modality names and signals.

    This class holds what is common to all the models:

    - :meth:`execute`, the entry point used by the Backend, which validates
      the signals before delegating to :meth:`integrate`.
    - The helpers to compute the synaptic connectivity (lateral and
      inter-areal), used by the neural architectures.

    Each concrete model (e.g. Cuppini2017, Kording2007) inherits from here
    and adds its own parameters, dynamics and causal-inference mechanism.
    A subclass must provide:

    - ``model_type``: ``"Neural"``, ``"Bayesian"`` or ``"MLE"``, so the
      Backend knows how to encode the stimuli.
    - :meth:`check_stimuli_compatible`: which modalities it accepts.
    - ``integrate(signals, **kwargs)``: the model itself. It returns a
      tuple ``(response, extra)``.
    - ``calculate_causes(**kwargs)``: the readout of the number of causes.
    """

    #: What kind of signal this model consumes. The Backend reads it to
    #: decide how the stimuli must be encoded. Subclasses override it.
    model_type = "Neural"

    def _calculate_lateral_synapses(
        self,
        neurons,
        excitation_loc,
        inhibition_loc,
        excitation_scale,
        inhibition_scale,
        dtype=np.float32,
    ):
        """
        Computes the values of lateral synapses.

        Designed for a group of recurrently connected neurons following a
        "Mexican hat" distribution (a central excitatory zone surrounded by
        an inhibitory annulus) calculated as a substraction of two Gaussians.
        For two different neurons at a distance :math:`d`, the synapse is

        .. math::
            L(d) = L_{ex} e^{-d^2 / 2 \\sigma_{ex}^2}
                 - L_{in} e^{-d^2 / 2 \\sigma_{in}^2}

        and a neuron has no synapse with itself. The distance wraps around,
        as the neurons are arranged in a ring.

        Parameters
        ----------
        neurons : int
            The number of neurons.
        excitation_loc: float
            Amplitude (:math:`L_{ex}`) of the excitatory Gaussian function.
            Despite its name, it is the height of the Gaussian and not its
            center, which is always the neuron itself.
        inhibition_loc: float
            Amplitude (:math:`L_{in}`) of the inhibitory Gaussian function.
        excitation_scale: float
            Width (:math:`\\sigma_{ex}`, in neurons) of the excitatory
            Gaussian function.
        inhibition_scale: float
            Width (:math:`\\sigma_{in}`, in neurons) of the inhibitory
            Gaussian function.
        dtype: numpy class
            Type of the array to store the values.

        Returns
        -------
        numpy.array
            The values of lateral synapses, a ``(neurons, neurons)`` array.
            The entry ``[i, j]`` is the synapse between neurons ``i`` and
            ``j``. As it only depends on their distance, it is symmetric.
        """
        the_lateral_synapses = np.zeros((neurons, neurons), dtype)

        for neuron_i in range(neurons):
            for neuron_j in range(neurons):
                # No self-connections.
                if neuron_i == neuron_j:
                    the_lateral_synapses[neuron_i, neuron_j] = 0
                    continue

                # Distance on the ring, then the two Gaussians centered on
                # the neuron: a narrow excitatory one and a wide inhibitory
                # one. Their difference is the Mexican hat.
                distance = calculate_neural_distance(neurons, neuron_i, neuron_j)
                e_gauss = excitation_loc * np.exp(
                    -(np.square(distance)) / (2 * np.square(excitation_scale))
                )
                i_gauss = inhibition_loc * np.exp(
                    -(np.square(distance)) / (2 * np.square(inhibition_scale))
                )

                the_lateral_synapses[neuron_i, neuron_j] = e_gauss - i_gauss
        return the_lateral_synapses

    def _calculate_inter_areal_synapses(
        self, neurons, weight, sigma, dtype=np.float32
    ):
        """
        Computes the values of inter-areal synapses.

        Designed for two connected group of neurons following a Gaussian
        function. It assumes symmetrical connectivity and the same number
        of neurons in both groups. The synapse between the neuron ``j`` of
        one group and the neuron ``k`` of the other one is

        .. math::
            W(d) = w \\, e^{-d^2 / 2 \\sigma^2}

        with :math:`d` the distance between them on the ring, so the
        strongest synapses connect neurons that encode the same position.

        Parameters
        ----------
        neurons : int
            The number of neurons in each group.
        weight: float
            The highest level of synaptic efficacy.
        sigma: float
            The width of the Gaussian function.
        dtype: numpy class
            Type of the array to store the values.

        Returns
        -------
        numpy.array
            The value of inter-areal synapses, a ``(neurons, neurons)``
            array. The entry ``[j, k]`` goes from the neuron ``k`` of the
            source group to the neuron ``j`` of the target group.
        """
        the_synapses = np.zeros((neurons, neurons), dtype=dtype)

        for j in range(neurons):
            for k in range(neurons):
                # The efficacy decays as a Gaussian of the distance
                # between the neurons (on the ring), peaking at `weight`.
                d = calculate_neural_distance(neurons, j, k)
                the_synapses[j, k] = weight * np.exp(
                    -(np.square(d)) / (2 * np.square(sigma))
                )
        return the_synapses

    def execute(self, signals, **kwargs):
        """
        Orchestrates the run over a set of signals.

        Validates modality-keyed sensory signals and delegates integration.

        Parameters
        ----------
        signals : Mapping
            The signals of the run, keyed by modality. Each key must be
            the ``modality`` of its signal.
        **kwargs
            The run parameters (time and position grids, random
            generator), forwarded as they are to :meth:`integrate`.

        Returns
        -------
        tuple
            ``(response, extra)`` as returned by :meth:`integrate`.

        Raises
        ------
        TypeError
            If ``signals`` is not a mapping.
        ValueError
            If a key does not match the modality of its signal, or if the
            modalities are not accepted by the model.
        """
        # The container must be keyed by modality...
        if not isinstance(signals, Mapping):
            raise TypeError("signals must be a mapping keyed by modality.")
        # ...and every key must agree with the modality the signal declares,
        # so a signal can never be routed to the wrong layer.
        for modality, signal in signals.items():
            if modality != signal.modality:
                raise ValueError(
                    f"Signal key {modality!r} does not match its modality "
                    f"{signal.modality!r}."
                )
        # The model decides which combination of modalities it accepts.
        self.check_stimuli_compatible(signals.keys())
        return self.integrate(signals, **kwargs)

    def check_stimuli_compatible(self, modalities):
        """
        Validates the modalities of the incoming stimuli.

        Receives modality names instead of Stimulus objects, so the
        integrator is not coupled to the stimulus descriptions. Each
        concrete architecture implements its own compatibility rules.

        Parameters
        ----------
        modalities : iterable of str
            Modality names of the incoming signals.

        Raises
        ------
        ValueError
            If the modalities are not accepted by the model.
        NotImplementedError
            In this base class: every subclass must override it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "check_stimuli_compatible()."
        )
