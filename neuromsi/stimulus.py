import numpy as np

from utils.neural_tools import calculate_neural_distance

from .signal import Signal


class Stimulus:
    """
    Represents a single external sensory stimulus, independent of any
    neural architecture that will later process it.

    It only describes "what happened in the world": where, how strong, how
    uncertain it is, and when. It also knows how to render itself as a
    neural input (``calculate_stimuli_input`` and
    ``create_unimodal_stimuli_matrix``), but it knows nothing about the
    integration dynamics or the connectivity of the network that will
    consume it.
    """

    #: Modality name. Backend uses it to route this stimulus to the
    #: corresponding integrator input (e.g. "visual", "auditory").
    #: Subclasses override it.
    modality: str = "generic"

    def __init__(
        self,
        position,
        intensity,
        sigma,
        onset=0,
        duration=None,
        stim_n=1,
        soa=None,
    ):
        """
        Parameters
        ----------
        position : float
            Spatial position of the stimulus, in degrees.
        intensity : float
            Maximum amplitude of the stimulus (E0 in the paper's notation).
        sigma : float
            Spatial spread/uncertainty of the stimulus (sigma_c in the
            paper). It describes the stimulus itself, not a neural
            receptive field.
        onset : float, optional
            Time (ms) at which the stimulus starts. Default 0.
        duration : float or None, optional
            How long the stimulus lasts (ms). If None, the integrator
            interprets it as "it lasts the whole simulation".
        stim_n : int, optional
            Number of repetitions of this stimulus over time (e.g. two
            beeps instead of one). Default 1.
        soa : float or None, optional
            Stimulus Onset Asynchrony: temporal separation (ms) between
            repetitions, only used if stim_n > 1.
        """
        self.position = position
        self.intensity = intensity
        self.sigma = sigma
        self.onset = onset
        self.duration = duration
        self.stim_n = stim_n
        self.soa = soa

    def __repr__(self):
        return (
            f"{type(self).__name__}(position={self.position}, "
            f"intensity={self.intensity}, sigma={self.sigma}, "
            f"onset={self.onset}, duration={self.duration}, "
            f"stim_n={self.stim_n}, soa={self.soa})"
        )

    def calculate_stimuli_input(self, neurons, dtype=np.float32):
        """
        Renders this stimulus as a spatial input Gaussian.

        Discretizes the stimulus (position, intensity and sigma) over an
        array of ``neurons`` neurons that encode the 1D space. The
        Gaussian reflects the uncertainty in the stimulus detection.

        Parameters
        ----------
        neurons : int
            Number of neurons over which the stimulus is sampled.
        dtype : numpy class, optional
            Data type of the resulting array. Default np.float32.

        Returns
        -------
        numpy.array
            The value of the stimulus for each neuron.
        """
        the_stimuli = np.zeros(neurons, dtype=dtype)

        for neuron_j in range(neurons):
            distance = calculate_neural_distance(neurons, neuron_j, self.position)
            the_stimuli[neuron_j] = self.intensity * np.exp(
                -(np.square(distance)) / (2 * np.square(self.sigma))
            )

        return the_stimuli

    def create_unimodal_stimuli_matrix(
        self, neurons, simulation_length, time_res, duration=None, dtype=np.float32
    ):
        """
        Builds the temporal input matrix of this stimulus.

        Combines the spatial input (``calculate_stimuli_input``) with the
        stimulus timing (onset, duration, repetitions and SOA) to produce
        the activation matrix (times x neurons) that the integrator
        consumes.

        Parameters
        ----------
        neurons : int
            Number of neurons over which the stimulus is sampled.
        simulation_length : float
            Total duration of the simulation in time units.
        time_res : float
            Temporal resolution of the simulation (integration step).
        duration : float or None, optional
            Effective duration of the stimulus. If None, the stimulus's
            own duration is used, and if that is also undefined, the
            stimulus is assumed to last the whole simulation.
        dtype : numpy class, optional
            Data type of the resulting array. Default np.float32.

        Returns
        -------
        numpy.array
            The input matrix (times x neurons).

        Raises
        ------
        ValueError
            If the total duration of all stimuli plus that of the
            inter-stimulus intervals exceeds ``simulation_length``.
        """
        onset = int(self.onset)
        stimuli_n = int(self.stim_n)
        soa = self.soa

        if duration is None:
            duration = self.duration if self.duration is not None else simulation_length

        if duration * stimuli_n > simulation_length:
            raise ValueError("Stimuli total duration exceeds simulation length.")

        if soa is not None:
            soa = int(soa)
            if duration * stimuli_n + soa * (stimuli_n - 1) > simulation_length:
                raise ValueError(
                    "Stimuli total duration exceeds simulation length."
                )
            if soa < duration:
                raise ValueError("SOA must be longer than stimulus duration.")

        stimuli = self.calculate_stimuli_input(neurons, dtype=dtype)
        no_stim = np.zeros(neurons, dtype=dtype)

        if stimuli_n == 0:
            stim = np.tile(no_stim, (simulation_length, 1))
            stimuli_matrix = np.repeat(stim, 1 / time_res, axis=0)
            return stimuli_matrix

        # Input before onset
        pre_stim = np.tile(no_stim, (onset, 1))

        # Input during stimulus delivery
        stim = np.tile(stimuli, (duration, 1))

        # Input during onset asynchrony
        soa_stim = (
            np.tile(no_stim, (soa - duration, 1))
            if soa is not None
            else None
        )

        # Input after stimulation
        post_stim_time = simulation_length - onset - duration * stimuli_n
        post_stim_time = (
            post_stim_time - (soa - duration) * (stimuli_n - 1)
            if soa is not None
            else post_stim_time
        )
        post_stim = np.tile(no_stim, (post_stim_time, 1))

        # Input concatenation
        stim_list = [stim, soa_stim] * (stimuli_n - 1)
        complete_stim = np.vstack((pre_stim, *stim_list, stim, post_stim))
        stimuli_matrix = np.repeat(complete_stim, 1 / time_res, axis=0)

        return stimuli_matrix

    def generate_signal(
        self,
        config,
        neurons,
        simulation_length,
        time_res,
        mode=0,
        dtype=np.float32,
    ):
        """
        Builds the complete Signal with which this stimulus enters the
        integrator: it creates the unimodal temporal matrix and packages
        it together with the synapses of its layer.

        Parameters
        ----------
        config : dict
            The dict returned by ``Integrator.get_conf()``, with the
            synapses of the whole architecture (lateral, cross-modal and
            feedforward).
        neurons : int
            Number of neurons of the integrator.
        simulation_length : float
            Total duration of the simulation.
        time_res : float
            Temporal resolution of the simulation.
        mode : int, optional
            Which unisensory layer of the integrator this stimulus belongs
            to: 0 (mode0) or 1 (mode1). Determines which synapses are
            associated with the Signal. Default 0.
        dtype : numpy class, optional
            Data type of the temporal matrix. Default np.float32.

        Returns
        -------
        Signal
            The signal ready to be integrated.
        """
        unimodal_matrix = self.create_unimodal_stimuli_matrix(
            neurons=neurons,
            simulation_length=simulation_length,
            time_res=time_res,
            dtype=dtype,
        )

        if mode == 0:
            latsynapses = config["mode0_latsynapses"]
            crossmodal_synapses = config["mode1_to_mode0_synapses"]
            feedforward_synapses = config["mode0_to_multi_synapses"]
        else:
            latsynapses = config["mode1_latsynapses"]
            crossmodal_synapses = config["mode0_to_mode1_synapses"]
            feedforward_synapses = config["mode1_to_multi_synapses"]

        return Signal(
            payload=self,
            metadata={
                "unimodal_matrix": unimodal_matrix,
                "latsynapses": latsynapses,
                "crossmodal_synapses": crossmodal_synapses,
                "feedforward_synapses": feedforward_synapses,
                "multi_latsynapses": config["multi_latsynapses"],
            },
        )

    @classmethod
    def get_stimuli(cls):
        """Returns an instance of the class with default values."""
        return cls(
            position=90,
            intensity=27,
            sigma=4,
            onset=0,
            duration=100,
        )


class Visual(Stimulus):
    """A visual stimulus (e.g. a flash of light at a position)."""

    modality = "visual"

    @classmethod
    def get_stimuli(cls):
        """Returns a visual stimulus with the default values."""
        return cls(
            position=90,
            intensity=27,
            sigma=4,
            onset=0,
            duration=100,
        )


class Audio(Stimulus):
    """An auditory stimulus (e.g. a beep at a position)."""

    modality = "auditory"

    @classmethod
    def get_stimuli(cls):
        """Returns an auditory stimulus with the default values."""
        return cls(
            position=100,
            intensity=28,
            sigma=32,
            onset=0,
            duration=100,
        )