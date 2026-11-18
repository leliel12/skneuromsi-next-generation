"""Backend: runs an experiment by connecting stimuli with an integrator."""

import numpy as np

from neuromsi.core import NDResult


class SignalDispatcher:
    """
    Links the stimuli of a run with the signals its integrator consumes.

    It picks, from the ``model_type`` of the integrator, how every stimulus
    must be encoded, and gathers the arguments that each kind of encoder
    needs. The stimuli only expose plain encoders and know nothing about
    the integrator; the Backend only asks for the signals.

    To support a new family of models, write its stimulus encoder (in
    :class:`~neuromsi.stimulus.Stimulus`), an adapter here and register it
    in ``_ENCODERS``.

    Parameters
    ----------
    backend : Backend
        The run the signals are for. The dispatcher reads from it the
        ``integrator`` (its ``model_type`` selects the encoder; its
        ``neurons``, ``n_sim`` and ``noise`` shape the signals), the
        ``time_range``, the ``time_res`` and the random generator
        ``random``. It is meant to be created inside ``Backend.run``, so
        it always sees the current values of the run.

    Raises
    ------
    ValueError
        If the integrator of the backend does not declare a known
        ``model_type``.
    """

    #: Which adapter encodes the stimuli of each ``model_type``. Neural
    #: models consume a time-by-neuron drive; statistical models (Bayesian
    #: and MLE) consume a measurement and its variance.
    _ENCODERS = {
        "Neural": "_neural_signal",
        "Bayesian": "_statistical_signal",
        "MLE": "_statistical_signal",
    }

    def __init__(self, backend):
        model_type = getattr(backend.integrator, "model_type", None)
        if model_type not in self._ENCODERS:
            known = ", ".join(repr(k) for k in self._ENCODERS)
            raise ValueError(
                f"Unknown model_type {model_type!r}. Expected one of "
                f"{known}."
            )
        self.backend = backend
        self._encoder = getattr(self, self._ENCODERS[model_type])

    def _neural_signal(self, stim):
        """
        Encodes a stimulus as the drive of a neural model.

        Takes the number of neurons of the integrator and the time grid.
        The random generator is not used, since the noise of a neural
        model is added by the integrator.

        Parameters
        ----------
        stim : Stimulus
            The stimulus to encode.

        Returns
        -------
        NeuralSignal
            A time-by-neuron matrix for the modality of the stimulus.
        """
        return stim.generate_neural_signal(
            neurons=self.backend.integrator.neurons,
            simulation_length=self.backend.time_range[1],
            time_res=self.backend.time_res,
        )

    def _statistical_signal(self, stim):
        """
        Encodes a stimulus as the measurement of a statistical model.

        Takes the number of simulations of the integrator (``n_sim``, 1
        if it has none, since MLE models may not simulate observers) and
        whether it is noisy (``noise``, False if it has no such
        attribute), plus the random generator.

        Parameters
        ----------
        stim : Stimulus
            The stimulus to encode.

        Returns
        -------
        UnisensorySignal
            The variance of the stimulus and the measurement drawn from it.
        """
        return stim.generate_unisensory_signal(
            n=getattr(self.backend.integrator, "n_sim", 1),
            noise=getattr(self.backend.integrator, "noise", False),
            random=self.backend.random,
        )

    def encode(self, stim):
        """
        Encodes a stimulus as the signal the integrator consumes.

        Parameters
        ----------
        stim : Stimulus
            The stimulus to encode.

        Returns
        -------
        NeuralSignal or UnisensorySignal
            The signal of the stimulus, of the kind that the
            ``model_type`` of the integrator requires.
        """
        return self._encoder(stim)


class Backend:
    """
    Connects a set of Stimulus objects with an Integrator, routing each
    stimulus to the named input that the integrator expects (e.g.
    "auditory", "visual"). It has no architecture parameters of its own:
    its only responsibility is to know "who is who" among the pieces it
    connects and to orchestrate the run.

    A run goes from the description of the world to the result of the
    model: the stimuli encode themselves as signals, the integrator turns
    the signals into a response, and the Backend packs everything in an
    :class:`~neuromsi.core.NDResult`. The same Backend works with neural,
    Bayesian and MLE models, as it reads the ``model_type`` of the
    integrator to decide how the stimuli must be encoded.

    Examples
    --------
    >>> from neuromsi import Audio, Visual, Backend, Cuppini2017
    >>> backend = Backend(
    ...     stimuli=[Audio(position=60), Visual(position=70)],
    ...     integrator=Cuppini2017(),
    ...     time_range=(0, 150),
    ...     time_res=0.5,
    ...     seed=42,
    ... )
    >>> result = backend.run()
    """

    def __init__(
        self,
        stimuli,
        integrator,
        *,
        seed=None,
        time_range=(0, 100),
        time_res=0.01,
        position_range=(0, 180),
        position_res=1,
        causes_kind="count",
        causes_dim="space",
        causes_peak_threshold=0.15,
        causes_peak_distance=None,
    ):
        """
        Parameters
        ----------
        stimuli : list of Stimulus
            The stimuli to connect, in any order.
        integrator : object
            Implements `.check_stimuli_compatible(...)`,
            `.execute(...)` and `.calculate_causes(...)`.
        position_range : (float, float), optional
            The (min, max) position covered by the integrator neurons, in
            degrees. It must describe exactly `integrator.neurons`
            positions at `position_res`, otherwise the result cannot be
            labelled. Default (0, 180).
        position_res : float, optional
            Angular step (deg) between two consecutive positions.
            Default 1.
        """
        self.stimuli = tuple(stimuli)
        self.integrator = integrator

        self.seed = seed
        self.time_range = time_range
        self.time_res = float(time_res)
        self.causes_kind = causes_kind
        self.causes_dim = causes_dim
        self.causes_peak_threshold = causes_peak_threshold
        self.causes_peak_distance = causes_peak_distance
        self.position_res = float(position_res)
        self.position_range = position_range

        self.random = np.random.default_rng(seed=seed)

        self._check_position_range()

    def _check_position_range(self):
        """
        Checks that 'position_range' covers the integrator neurons.

        Every neuron encodes a slice of the space defined by 'position_range',
        so the number of neurons must be exactly the number of positions the
        range describes at 'position_res'. Checking it here avoids running
        the whole simulation only to fail when building the result.

        Models without neurons (e.g. statistical models, whose grid is
        defined by 'position_range' and 'position_res' alone) are skipped;
        NDResult validates the grid when the result is built.

        Raises
        ------
        ValueError
            If 'position_range' is not a (min, max) pair, or if it does not
            match the number of neurons of the integrator.

        """
        prange = self.position_range
        if len(prange) != 2 or prange[0] > prange[1]:
            raise ValueError(
                f"The position_range must be (min, max). Got {prange}"
            )

        neurons = getattr(self.integrator, "neurons", None)
        if neurons is None:
            return

        expected = int(abs(prange[1] - prange[0]) / self.position_res) or 1
        if expected != neurons:
            raise ValueError(
                "The position_range and position_res do not match the "
                f"integrator. Expected {expected} positions, "
                f"but the integrator has {neurons} neurons. "
                f"Set position_range=(0, {neurons * self.position_res})"
            )

    def _make_ndresult(self, response, extra, stimuli):
        """
        Builds the NDResult of the run out of the integrator response.

        This is where everything that is not the simulation itself happens:
        the causes are calculated, the response is labelled with the real
        modality of each stimulus, the run parameters are collected, and
        the whole thing is packed into an NDResult.

        Parameters
        ----------
        response : dict
            The raw response of the integrator, keyed by modality and
            ``multi``.
        extra : dict
            Extra information reported by the integrator. It is updated in
            place with the causes parameters used for this run.
        stimuli : tuple of Stimulus
            The original stimuli, used to label the result and record the
            experimental parameters.

        Returns
        -------
        NDResult
            The result of the run.
        """
        extra.update(
            {
                "causes_kind": self.causes_kind,
                "causes_dim": self.causes_dim,
                "causes_peak_threshold": self.causes_peak_threshold,
                "causes_peak_distance": self.causes_peak_distance,
            }
        )

        # Keep each unisensory response under its modality name.
        modes = {stim.modality: response[stim.modality] for stim in stimuli}
        modes["multi"] = response["multi"]

        causes = self._calculate_causes(response, extra)
        run_parameters = {
            "seed": self.seed,
            "time_range": self.time_range,
            "time_res": self.time_res,
            "position_range": self.position_range,
            "position_res": self.position_res,
            "stimuli": {
                stim.modality: {
                    "position": stim.position,
                    "intensity": stim.intensity,
                    "sigma": stim.sigma,
                    "onset": stim.onset,
                    "duration": stim.duration,
                    "stim_n": stim.stim_n,
                    "soa": stim.soa,
                }
                for stim in stimuli
            },
        }

        # The architecture parameters are part of the experiment: without
        # them the result cannot be reproduced.
        model_parameters = getattr(self.integrator, "model_parameters", None)
        if model_parameters:
            run_parameters["model"] = dict(model_parameters)

        return NDResult.from_modes_dict(
            mname=type(self.integrator).__name__,
            mtype=self.integrator.model_type,
            output_mode="multi",
            nmap={mode: mode for mode in modes},
            modes_dict=modes,
            time_range=self.time_range,
            position_range=self.position_range,
            time_res=self.time_res,
            position_res=self.position_res,
            causes=causes,
            run_parameters=run_parameters,
            extra=extra,
        )

    def _calculate_causes(self, response, extra):
        """Shortcut to call integrator.calculate_causes(...)."""
        return self.integrator.calculate_causes(
            multi=response["multi"], **extra
        )

    def run(self):
        """
        Runs the experiment and returns its result.

        The run goes through four steps:

        1. The integrator checks that the modalities of the stimuli are
           compatible with the model.
        2. Each stimulus is encoded as the signal the model consumes (a
           time-by-neuron matrix for neural models, a variance and a
           measurement for statistical ones).
        3. The integrator processes the signals over the configured time
           and position grids.
        4. The response is packed, together with the causes, the run
           parameters and the position of each stimulus, into an NDResult.

        Returns
        -------
        NDResult
            The result of the run, with one response per modality plus the
            multisensory one (``multi``).

        Raises
        ------
        ValueError
            If the stimuli are not compatible with the integrator, or if
            the integrator does not declare a known ``model_type``.

        """
        # 1. Fail early, before encoding anything, if the model cannot
        #    handle the modalities of the stimuli.
        self.integrator.check_stimuli_compatible(
            [stim.modality for stim in self.stimuli]
        )

        # 2. Encode each stimulus as the signal this model consumes,
        #    keyed by modality.
        dispatcher = SignalDispatcher(self)
        signals = {}
        for stim in self.stimuli:
            signals[stim.modality] = dispatcher.encode(stim)

        # 3. Integrate the signals. The backend shares its random generator
        #    so the whole run is reproducible from a single seed.
        response, extra = self.integrator.execute(
            signals,
            time_range=self.time_range,
            time_res=self.time_res,
            position_range=self.position_range,
            position_res=self.position_res,
            random=self.random,
        )

        # Keep experiment descriptors outside the encoded sensory signals.
        extra["stim_position"] = [stim.position for stim in self.stimuli]

        # 4. Pack the response, causes and run parameters into the result.
        ndresult = self._make_ndresult(response, extra, self.stimuli)

        return ndresult
