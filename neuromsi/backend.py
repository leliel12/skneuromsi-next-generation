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

    # Preferred input order, keyed by integrator class name. Extend this
    # dict when adding new architectures; integrators not listed here
    # keep the stimuli in the order they were given.
    expected_input_order = {
        "Cuppini2017": ("auditory", "visual"),
    }

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
            The stimuli to connect, in any order. ``order_stims()``
            reorders them into the input order the integrator expects.
        integrator : object
            Implements `.get_conf()`, `.check_stimuli_compatible(...)`,
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

    def order_stims(self):
        """
        Reorders the stimuli into the input order the integrator expects.

        Looks up ``expected_input_order`` by the integrator class name. If
        the integrator is not listed there, or a listed modality is
        missing from the stimuli, the original order is kept.
        """
        order = self.expected_input_order.get(type(self.integrator).__name__)
        if order is None:
            return self.stimuli

        by_modality = {stim.modality: stim for stim in self.stimuli}
        try:
            return tuple(by_modality[modality] for modality in order)
        except KeyError:
            return self.stimuli

    def _check_position_range(self):
        """
        Checks that 'position_range' covers the integrator neurons.

        Every neuron encodes a slice of the space defined by 'position_range',
        so the number of neurons must be exactly the number of positions the
        range describes at 'position_res'. Checking it here avoids running
        the whole simulation only to fail when building the result.

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

        neurons = self.integrator.neurons
        expected = int(abs(prange[1] - prange[0]) / self.position_res) or 1
        if expected != neurons:
            raise ValueError(
                "The position_range and position_res do not match the "
                f"integrator. Expected {expected} positions, "
                f"but the integrator has {neurons} neurons. "
                f"Set position_range=(0, {neurons * self.position_res})"
            )

    def run(self):
        """
        Runs the simulation and returns its result.

        Validates the stimuli, asks the integrator for its configuration,
        generates a signal per stimulus, runs the integration, and delegates
        the packaging of the result to ``make_ndresult``.
        """

        self.integrator.check_stimuli_compatible(
            [stim.modality for stim in self.stimuli]
        )
        
        stimuli = self.order_stims()
        intg_conf = self.integrator.get_conf()

        signals = []
        for idx, stim in enumerate(stimuli):
            signals.append(
                stim.generate_signal(
                    config=intg_conf,
                    mode=idx,
                    neurons=self.integrator.neurons,
                    simulation_length=self.time_range[1],
                    time_res=self.time_res,
                )
            )

        response, extra = self.integrator.execute(
            *signals,
            time_range=self.time_range,
            time_res=self.time_res,
            random=self.random,
        )

        return self.make_ndresult(response, extra, stimuli)

    def make_ndresult(self, response, extra, stimuli):
        """
        Builds the NDResult of the run out of the integrator response.

        This is where everything that is not the simulation itself happens:
        the causes are calculated, the response is labelled with the real
        modality of each stimulus, the run parameters are collected, and
        the whole thing is packed into an NDResult.

        Parameters
        ----------
        response : dict
            The raw response of the integrator, keyed by 'mode0', 'mode1',
            'multi', etc.
        extra : dict
            Extra information reported by the integrator. It is updated in
            place with the causes parameters used for this run.
        stimuli : tuple of Stimulus
            The stimuli in the same order in which they were given to the
            integrator, i.e. the 'idx'-th one produced the 'mode{idx}'
            entry of 'response'.

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

        # replace the generic mode names of the integrator by the real
        # modality of the stimulus that fed each unisensory layer
        modes = {
            stim.modality: response[f"mode{idx}"]
            for idx, stim in enumerate(stimuli)
        }
        modes["multi"] = response["multi"]

        causes = self.calculate_causes(response, extra)
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

        return NDResult.from_modes_dict(
            mname=type(self.integrator).__name__,
            mtype="integrator",
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

    def calculate_causes(self, response, extra):
        """Shortcut to call integrator.calculate_causes(...)."""
        return self.integrator.calculate_causes(multi=response["multi"], **extra)
