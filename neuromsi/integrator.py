import numpy as np

from utils.neural_tools import calculate_neural_distance


class Integrator:
    """
    Base class for any neural architecture.

    Holds what is common to all of them: the methods to compute the
    synaptic connectivity (lateral and inter-areal). Each concrete
    architecture (e.g. Cuppini2017) inherits from here and adds its own
    parameters, dynamics and causal-inference mechanism.
    """

    def calculate_lateral_synapses(
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

        Parameters
        ----------
        neurons : int
            The number of neurons.
        excitation_loc: float
            Loc of the excitatory Gaussian function.
        inhibition_loc: float
            Loc of the inhibitory Gaussian function.
        excitation_scale: float
            Scale of the excitatory Gaussian function.
        inhibition_scale: float
            Scale of the inhibitory Gaussian function.
        dtype: numpy class
            Type of the array to store the values.

        Returns
        -------
        numpy.array
            The values of lateral synapses.
        """
        the_lateral_synapses = np.zeros((neurons, neurons), dtype)

        for neuron_i in range(neurons):
            for neuron_j in range(neurons):
                if neuron_i == neuron_j:
                    the_lateral_synapses[neuron_i, neuron_j] = 0
                    continue

                distance = calculate_neural_distance(neurons, neuron_i, neuron_j)
                e_gauss = excitation_loc * np.exp(
                    -(np.square(distance)) / (2 * np.square(excitation_scale))
                )
                i_gauss = inhibition_loc * np.exp(
                    -(np.square(distance)) / (2 * np.square(inhibition_scale))
                )

                the_lateral_synapses[neuron_i, neuron_j] = e_gauss - i_gauss
        return the_lateral_synapses

    def calculate_inter_areal_synapses(
        self, neurons, weight, sigma, dtype=np.float32
    ):
        """
        Computes the values of inter-areal synapses.

        Designed for two connected group of neurons following a Gaussian
        function. It assumes symmetrical connectivity and the same number
        of neurons in both groups.

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
            The value of inter-areal synapses.
        """
        the_synapses = np.zeros((neurons, neurons), dtype=dtype)

        for j in range(neurons):
            for k in range(neurons):
                d = calculate_neural_distance(neurons, j, k)
                the_synapses[j, k] = weight * np.exp(
                    -(np.square(d)) / (2 * np.square(sigma))
                )
        return the_synapses

    def execute(self, *signals, **kwargs):
        """
        Orchestrates the run over a set of signals.

        Validates the modalities of the signals (via
        ``check_stimuli_compatible``) and delegates the integration to
        ``integrate``, passing the simulation ``**kwargs`` along (e.g.
        time_range, time_res, random). It does not assume how many
        signals or stimuli there are.
        """
        modalities = [signal.payload.modality for signal in signals]
        self.check_stimuli_compatible(modalities)
        return self.integrate(*signals, **kwargs)

    def check_stimuli_compatible(self, modalities):
        """
        Validates the modalities of the incoming stimuli.

        Receives an iterable of ``modality`` strings (e.g. "visual",
        "auditory") instead of Stimulus objects, so the integrator is not
        coupled to the stimuli. Each concrete architecture implements its
        own rules (how many modalities it accepts and which ones).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "check_stimuli_compatible()."
        )