import numpy as np


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

        self.random = np.random.default_rng(seed=seed)

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

    def run(self):
        """Runs the simulation, delegating all the work to the integrator."""

        self.integrator.check_stimuli_compatible(
            [stim.modality for stim in stimuli]
        )
        
        stimuli = self.order_stims()

        intg_conf = self.integrator.get_conf()

        signals = []
        for idx, stim in enumerate(stimuli):
            signals.append(
                stim.generate_signal(
                    config=intg_conf,
                    mode=idx + 1,
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

        extra.update({
            "causes_kind": self.causes_kind,
            "causes_dim": self.causes_dim,
            "causes_peak_threshold": self.causes_peak_threshold,
            "causes_peak_distance": self.causes_peak_distance,
        })

        _res = {
            stim.modality: response[f"mode_{idx + 1}"]
            for idx, stim in enumerate(stimuli)
        }
        _res["multi"] = response["multi"]

        return _res, extra

    def calculate_causes(self, response, extra):
        """Shortcut to call integrator.calculate_causes(...)."""
        return self.integrator.calculate_causes(multi=response["multi"], **extra)