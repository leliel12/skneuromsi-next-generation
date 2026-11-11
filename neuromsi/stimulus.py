"""Stimuli: descriptions of the external events that reach the senses.

A :class:`Stimulus` says what happened in the world (where, how strong, how
uncertain and when) and knows how to encode itself as the signal that a
model consumes (see :mod:`neuromsi.signal`). :class:`Visual` and
:class:`Audio` are the ready-to-use stimuli, with the default values of
each modality.
"""

import numpy as np

from neuromsi.utils.neural_tools import calculate_neural_distance

from .signal import NeuralSignal, UnisensorySignal


class Stimulus:
    """
    Represents a single external sensory stimulus, independent of any
    neural architecture that will later process it.

    It only describes "what happened in the world": where, how strong, how
    uncertain it is, and when. It also knows how to render itself as a
    neural input (``_calculate_stimuli_input`` and
    ``_create_unimodal_stimuli_matrix``), but it knows nothing about the
    integration dynamics or the connectivity of the network that will
    consume it.

    The encoding depends on the kind of model that consumes the stimulus:
    :meth:`generate_neural_signal` for neural models and
    :meth:`generate_unisensory_signal` for statistical models (Bayesian
    and MLE). The Backend calls the right one, so users only need to
    describe the stimulus.

    Attributes
    ----------
    position, intensity, sigma, onset, duration, stim_n, soa
        The values given to the constructor (see ``__init__``).
    modality : str
        Class attribute with the name of the sensory modality.
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
            How long the stimulus lasts (ms). If None, it lasts the whole
            simulation.
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

    def _calculate_stimuli_input(self, neurons, dtype=np.float32):
        """
        Renders this stimulus as a spatial input Gaussian.

        Discretizes the stimulus (position, intensity and sigma) over an
        array of ``neurons`` neurons that encode the 1D space. The
        Gaussian reflects the uncertainty in the stimulus detection:

        .. math::
            E(j) = E_0 \\, e^{-d(j, p)^2 / 2 \\sigma^2}

        where :math:`p` is the position of the stimulus, :math:`d` the
        distance on the ring of neurons, :math:`E_0` the intensity and
        :math:`\\sigma` the sigma of the stimulus. The position is
        used as a neuron index, so it assumes one neuron per position unit.

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
            # Distance from this neuron to the stimulus, on the ring (the
            # space wraps around). The closer the neuron, the stronger its
            # input; it peaks at `intensity` on the stimulus position.
            distance = calculate_neural_distance(
                neurons, neuron_j, self.position
            )
            the_stimuli[neuron_j] = self.intensity * np.exp(
                -(np.square(distance)) / (2 * np.square(self.sigma))
            )

        return the_stimuli

    def _create_unimodal_stimuli_matrix(
        self,
        neurons,
        simulation_length,
        time_res,
        dtype=np.float32,
    ):
        """
        Builds the temporal input matrix of this stimulus.

        Combines the spatial input (``_calculate_stimuli_input``) with the
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
            inter-stimulus intervals exceeds ``simulation_length``, if the
            SOA is shorter than the duration, or if the onset leaves no
            room for the whole stimulus inside the simulation (NumPy then
            reports "negative dimensions are not allowed").
        """
        # The matrix is built with one row per time unit (ms), so the
        # timing values are used as whole numbers. The resolution is
        # applied at the very end.
        onset = int(self.onset)
        stimuli_n = int(self.stim_n)
        soa = self.soa

        duration = (
            self.duration if self.duration is not None else simulation_length
        )

        # Validate that the whole sequence fits in the simulation.
        if duration * stimuli_n > simulation_length:
            raise ValueError(
                "Stimuli total duration exceeds simulation length."
            )

        if soa is not None:
            soa = int(soa)
            if (
                duration * stimuli_n + soa * (stimuli_n - 1)
                > simulation_length
            ):
                raise ValueError(
                    "Stimuli total duration exceeds simulation length."
                )
            if soa < duration:
                raise ValueError("SOA must be longer than stimulus duration.")

        # `stimuli` is the spatial profile while the stimulus is on, and
        # `no_stim` is the silence (all zeros) before, between and after.
        stimuli = self._calculate_stimuli_input(neurons, dtype=dtype)
        no_stim = np.zeros(neurons, dtype=dtype)

        # Nothing to deliver: the whole simulation is silent.
        if stimuli_n == 0:
            stim = np.tile(no_stim, (simulation_length, 1))
            stimuli_matrix = np.repeat(stim, 1 / time_res, axis=0)
            return stimuli_matrix

        # The timeline is, in order:
        #   silence (onset) | stimulus | gap | stimulus | ... | silence
        # Every block is a (time units, neurons) array, joined at the end.

        # Input before onset
        pre_stim = np.tile(no_stim, (onset, 1))

        # Input during stimulus delivery
        stim = np.tile(stimuli, (duration, 1))

        # Input during onset asynchrony: the gap between two repetitions
        # lasts SOA - duration, since SOA is measured from onset to onset.
        soa_stim = (
            np.tile(no_stim, (soa - duration, 1)) if soa is not None else None
        )

        # Input after stimulation: whatever time is left of the simulation.
        post_stim_time = simulation_length - onset - duration * stimuli_n
        post_stim_time = (
            post_stim_time - (soa - duration) * (stimuli_n - 1)
            if soa is not None
            else post_stim_time
        )
        post_stim = np.tile(no_stim, (post_stim_time, 1))

        # Input concatenation: (stimulus, gap) repeated stim_n - 1 times,
        # then the last stimulus, which has no gap after it.
        stim_list = [stim, soa_stim] * (stimuli_n - 1)
        complete_stim = np.vstack((pre_stim, *stim_list, stim, post_stim))
        # Refine the time axis: each time unit becomes 1 / time_res rows,
        # so the matrix has one row per integration step.
        stimuli_matrix = np.repeat(complete_stim, 1 / time_res, axis=0)

        return stimuli_matrix

    def _draw_measurement(self, *, noise, n, random):
        """Draws one unisensory measurement of this stimulus.

        This is the measurement a statistical model (Bayesian or MLE)
        works with: the true stimulus position plus a Gaussian draw whose
        scale is the stimulus uncertainty. It belongs to the stimulus
        because it only depends on where the stimulus is and how uncertain
        it is; ``noise``, the number of simulations and the random
        generator come from the model / backend that asks for it.

        Parameters
        ----------
        noise : bool
            Whether to include noise in the computation. If False, the
            true stimulus position is returned (a perfect measurement).
        n : int
            Number of simulations (one independent draw each).
        random : numpy.random.Generator
            The random number generator used for the draws.

        Returns
        -------
        float or numpy.ndarray
            The computed input: a scalar when ``noise`` is False,
            an array of ``n`` values otherwise.
        """
        # unisensory_var = np.square(self.sigma)
        # Without noise the measurement is the true position of the
        # stimulus: a perfect observer. We used to return
        # ``position + sqrt(var)`` (a fixed one-sigma offset), but that
        # is not a noise-free measurement: it biases every model that
        # consumes the signal (e.g. Kording2007 and AlaisBurr2004). The
        # offset had no theoretical justification, so it was removed.
        #
        # Previous behavior, kept for reference:
        # if noise is False:
        #     return self.position + np.sqrt(unisensory_var)
        if noise is False:
            return self.position
        # With noise: position + sigma * N(0, 1), one independent draw
        # for each simulation.
        return self.position + self.sigma * random.standard_normal(n)

    def generate_neural_signal(
        self,
        neurons,
        simulation_length,
        time_res,
        dtype=np.float32,
    ):
        """
        Encodes this stimulus as external sensory drive for an integrator.

        Parameters
        ----------
        neurons : int
            Number of neurons of the integrator.
        simulation_length : float
            Total duration of the simulation.
        time_res : float
            Temporal resolution of the simulation.
        dtype : numpy class, optional
            Data type of the temporal matrix. Default np.float32.

        Returns
        -------
        NeuralSignal
            The signal ready to be integrated.
        """
        values = self._create_unimodal_stimuli_matrix(
            neurons=neurons,
            simulation_length=simulation_length,
            time_res=time_res,
            dtype=dtype,
        )

        return NeuralSignal(modality=self.modality, values=values)

    def generate_unisensory_signal(self, n, noise, random):
        """
        Encodes this stimulus as a unisensory estimate for a model.

        Instead of a time-by-neuron matrix, statistical models (Bayesian
        and MLE) consume the stimulus variance and the unisensory input
        drawn from it (``_draw_measurement``).

        Parameters
        ----------
        n : int
            Number of simulations of the model.
        noise : bool
            Whether to include noise in the input computation. If False,
            the input is the true position of the stimulus.
        random : numpy.random.Generator
            The random number generator used for the draws.

        Returns
        -------
        UnisensorySignal
            The unisensory estimate ready to be used by the model.
        """
        values = {
            "var": float(np.square(self.sigma)),
            "input": self._draw_measurement(noise=noise, n=n, random=random),
        }

        return UnisensorySignal(modality=self.modality, values=values)


class Visual(Stimulus):
    """A visual stimulus (e.g. a flash of light at a position).

    Same as :class:`Stimulus`, with the default values of vision: a strong
    stimulus (``intensity=27``) with a precise location (``sigma=4``),
    lasting 100 ms. Only ``position`` is positional; everything else is
    keyword-only and can be overridden.

    Parameters
    ----------
    position : float
        Spatial position of the stimulus, in degrees.
    intensity : float, optional
        Maximum amplitude. Default 27.
    sigma : float, optional
        Spatial uncertainty. Default 4.
    onset : float, optional
        Start time (ms). Default 0.
    duration : float or None, optional
        Duration (ms). Default 100.
    **kwargs
        ``stim_n`` and ``soa``, as in :class:`Stimulus`.
    """

    modality = "visual"

    def __init__(
        self,
        position,
        *,
        intensity=27,
        sigma=4,
        onset=0,
        duration=100,
        **kwargs,
    ):
        super().__init__(
            position,
            intensity=intensity,
            sigma=sigma,
            onset=onset,
            duration=duration,
            **kwargs,
        )


class Audio(Stimulus):
    """An auditory stimulus (e.g. a beep at a position).

    Same as :class:`Stimulus`, with the default values of audition: a
    strong stimulus (``intensity=28``) with a much vaguer location than
    vision (``sigma=32``), lasting 100 ms. Only ``position`` is positional;
    everything else is keyword-only and can be overridden.

    Parameters
    ----------
    position : float
        Spatial position of the stimulus, in degrees.
    intensity : float, optional
        Maximum amplitude. Default 28.
    sigma : float, optional
        Spatial uncertainty. Default 32.
    onset : float, optional
        Start time (ms). Default 0.
    duration : float or None, optional
        Duration (ms). Default 100.
    **kwargs
        ``stim_n`` and ``soa``, as in :class:`Stimulus`.
    """

    modality = "auditory"

    def __init__(
        self,
        position,
        *,
        intensity=28,
        sigma=32,
        onset=0,
        duration=100,
        **kwargs,
    ):
        super().__init__(
            position,
            intensity=intensity,
            sigma=sigma,
            onset=onset,
            duration=duration,
            **kwargs,
        )
