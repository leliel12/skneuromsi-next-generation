import numpy as np

from neuromsi.core import NDResult


class Backend:
    """
    Connects a set of Stimulus objects with an Integrator, routing each
    stimulus to the named input that the integrator expects (e.g.
    "auditory", "visual"). It has no architecture parameters of its own:
    its only responsibility is to know "who is who" among the pieces it
    connects and to orchestrate the run.
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

        Models without neurons (e.g. Bayesian models, whose grid is defined
        by 'position_range' and 'position_res' alone) are skipped; NDResult
        validates the grid when the result is built.

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

        The model type decides the encoding: neural models receive a
        time-by-neuron matrix per modality, Bayesian models receive the
        stimulus variance and the unisensory measurement drawn from it.

        Returns
        -------
        dict
            One signal per stimulus, keyed by modality.

        Raises
        ------
        ValueError
            If the model does not declare a known ``model_type``.

        """
        model_type = getattr(self.integrator, "model_type", None)

        if model_type == "Bayesian":
            return {
                stim.modality: stim.generate_bayesian_signal(
                    n=self.integrator.n_sim,
                    noise=self.integrator.noise,
                    random=self.random,
                )
                for stim in self.stimuli
            }

        if model_type == "Neural":
            return {
                stim.modality: stim.generate_neural_signal(
                    neurons=self.integrator.neurons,
                    simulation_length=self.time_range[1],
                    time_res=self.time_res,
                )
                for stim in self.stimuli
            }

        raise ValueError(
            f"Unknown model_type {model_type!r}. Expected 'Neural' or "
            f"'Bayesian'. Got {model_type!r}."
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
        extra.update({
            "causes_kind": self.causes_kind,
            "causes_dim": self.causes_dim,
            "causes_peak_threshold": self.causes_peak_threshold,
            "causes_peak_distance": self.causes_peak_distance,
        })

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
        Runs the simulation and returns its result.

        Validates stimuli, encodes one sensory signal per modality, runs
        the integration, and delegates result packaging to ``_make_ndresult``.
        """

        self.integrator.check_stimuli_compatible(
            [stim.modality for stim in self.stimuli]
        )

        signals = self._generate_signals()

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

        return self._make_ndresult(response, extra, self.stimuli)
