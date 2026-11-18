"""Backend: runs an experiment by connecting stimuli with an integrator."""

import numpy as np

from neuromsi.core import NDResult


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

    def _generate_signals(self):
        """
        Encodes every stimulus as the signal its model consumes.

        The Backend does not know how a signal is built: it hands itself
        to each stimulus, which reads from it the integrator and the
        context of the run and decides the encoding (see
        :meth:`Stimulus.generate_signal`).

        Returns
        -------
        dict
            One signal per stimulus, keyed by modality.

        Raises
        ------
        ValueError
            If the model does not declare a known ``model_type``.

        """
        signals = {}
        for stim in self.stimuli:
            signals[stim.modality] = stim.generate_signal(self)
        return signals

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
        signals = self._generate_signals()

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
