import copy
import numpy as np

from dataclasses import dataclass



from utils.neural_tools import (
    calculate_neural_distance,
    calculate_stimuli_input,
    create_unimodal_stimuli_matrix,
)
from utils.readout_tools import calculate_spatiotemporal_causes_from_peaks


# =========================================================================
# 1. STIMULI: represent ONLY the external physical event
# =========================================================================

class Stimulus:
    """
    Represents a single external sensory stimulus, independent of any
    neural architecture that will later process it.

    It only describes "what happened in the world": where, how strong, how
    uncertain and when. It knows nothing about neurons, receptive fields
    or integration dynamics.
    """

    #: Modality name. Wiring uses it to route this stimulus to the
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


class Visual(Stimulus):
    """A visual stimulus (e.g. a flash of light at a position)."""

    modality = "visual"


class Audio(Stimulus):
    """An auditory stimulus (e.g. a beep at a position)."""

    modality = "auditory"

@dataclass
class Signal:
    """
    Packages ONE stimulus (overload) together with all the values that
    belong to that modality: duration, temporal matrix, point input,
    lateral synapses of its layer, the cross-modal synapses it receives
    from the other layer, and the feedforward synapses toward the
    multisensory layer.

    It also carries a copy of the shared simulation values (n_time_steps,
    hist_times, multi_latsynapses) so that each Signal is self-contained.
    Wiring builds one per stimulus and passes them to integrate().
    """

    name: str ## needed?
    overload: Stimulus # same, in fact so much info ends up being unnecessary
    
    unimodal_matrix: np.ndarray
    latsynapses: np.ndarray
    crossmodal_synapses: np.ndarray
    feedforward_synapses: np.ndarray
    multi_latsynapses: np.ndarray ## odd


# =========================================================================
# 2. INTEGRATOR: generic base + Cuppini (2017) architecture
# =========================================================================

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


class Cuppini2017(Integrator):
    """
    Neural integrator implementing the Cuppini et al. (2017) architecture:
    two reciprocally connected unisensory layers (auditory, visual) that
    feed a third multisensory layer ("causal inference area").

    Every parameter here is specific to THIS architecture: number of
    neurons, sigmoid dynamics, lateral connectivity (Mexican Hat), and
    the strength of the cross-modal / feedforward synapses. A different
    integrator could use completely different parameters, or a different
    mechanism, and still accept the same Stimulus objects.
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
        seed=None,
        position_range=(0, 180),
        position_res=1,
        time_range=(0, 100),
        time_res=0.01,
        causes_kind="count",
        causes_dim="space",
        causes_peak_threshold=0.15,
        causes_peak_distance=None,
    ):
        if len(tau) != 3:
            raise ValueError(
                "tau must have exactly 3 values: "
                "(mode0, mode1, multisensory)"
            )

        self.neurons = neurons
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
        self.position_range = position_range
        self.position_res = float(position_res)
        self.time_range = time_range
        self.time_res = float(time_res)

        self.causes_kind = causes_kind
        self.causes_dim = causes_dim
        self.causes_peak_threshold = causes_peak_threshold
        self.causes_peak_distance = causes_peak_distance

        self.random = np.random.default_rng(seed=seed)

    def sigmoid(self, u):
        """Sigmoid activation function F(u), shared by the 3 layers."""
        return 1 / (1 + np.exp(-self.s * (u - self.theta)))

    def get_synaps_config(self):

        # --- Lateral connectivity (Mexican Hat), within each layer ---
        mode0_latsynapses = self.calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["mode0"]
        )
        mode1_latsynapses = self.calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["mode1"]
        )
        multi_latsynapses = self.calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["multi"]
        )

        # --- Cross-modal synapses (Wav, Wva) ---
        mode0_to_mode1_synapses = self.calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.cross_modal_weight,
            sigma=self.cross_modal_sigma,
        )
        mode1_to_mode0_synapses = self.calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.cross_modal_weight,
            sigma=self.cross_modal_sigma,
        )

        # --- Feedforward synapses to the multisensory layer (Wma, Wmv) ---
        mode0_to_multi_synapses = self.calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.feedforward_weight,
            sigma=self.feedforward_sigma,
        )
        mode1_to_multi_synapses = self.calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.feedforward_weight,
            sigma=self.feedforward_sigma,
        )

        return {
            "mode0_latsynapses": mode0_latsynapses,
            "mode1_latsynapses": mode1_latsynapses,
            "multi_latsynapses": multi_latsynapses,
            "mode0_to_mode1_synapses": mode0_to_mode1_synapses,
            "mode1_to_mode0_synapses": mode1_to_mode0_synapses,
            "mode0_to_multi_synapses": mode0_to_multi_synapses,
            "mode1_to_multi_synapses": mode1_to_multi_synapses,
        }
        
    def integrate(self, signal_1, signal_2):

        hist_times = np.arange(
                            self.time_range[0], self.time_range[1], self.time_res
                        )
        n_time_steps = hist_times.size

        # --- State containers ---
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
            # Cross-modal input (via Wva, Wav)
            mode0_cm_input = np.sum(
                signal_1.crossmodal_synapses * mode1_y, axis=1
            )
            mode1_cm_input = np.sum(
                signal_2.crossmodal_synapses * mode0_y, axis=1
            )

            # Feedforward input to the multisensory layer
            multi_input = (
                np.sum(signal_1.feedforward_synapses * mode0_y, axis=1)
                + np.sum(signal_2.feedforward_synapses * mode1_y, axis=1)
            )

            # External (stimulus) + cross-modal input
            mode0_input = signal_1.unimodal_matrix[i] + mode0_cm_input
            mode1_input = signal_2.unimodal_matrix[i] + mode1_cm_input

            # Noise, if enabled
            if self.noise:
                mode0_noise = -(
                    signal_1.overload.intensity * self.noise_level
                ) + (
                    2 * signal_1.overload.intensity * self.noise_level
                ) * self.random.random(self.neurons)
                mode1_noise = -(
                    signal_2.overload.intensity * self.noise_level
                ) + (
                    2 * signal_2.overload.intensity * self.noise_level
                ) * self.random.random(self.neurons)
                mode0_input = mode0_input + mode0_noise
                mode1_input = mode1_input + mode1_noise

            # Lateral input (Mexican Hat), within each layer
            la = np.sum(signal_1.latsynapses * mode0_y, axis=1)
            lv = np.sum(signal_2.latsynapses * mode1_y, axis=1)
            lm = np.sum(signal_1.multi_latsynapses * multi_y, axis=1)

            # Total input per layer
            u_a = la + mode0_input
            u_v = lv + mode1_input
            u_m = lm + multi_input

            # Euler step: y_new = y + dt * (1/tau) * (-y + sigmoid(u))
            mode0_y = mode0_y + self.time_res * (
                (-mode0_y + self.sigmoid(u_a)) / tau_mode0
            )
            mode1_y = mode1_y + self.time_res * (
                (-mode1_y + self.sigmoid(u_v)) / tau_mode1
            )
            multi_y = multi_y + self.time_res * (
                (-multi_y + self.sigmoid(u_m)) / tau_multi
            )

            mode0_res[i, :] = mode0_y
            mode1_res[i, :] = mode1_y
            multi_res[i, :] = multi_y

        response = {
            "mode0": mode0_res,
            "mode1": mode1_res,
            "multi": multi_res,
        }
        extra = {
            "causes_kind": self.causes_kind,
            "causes_dim": self.causes_dim,
            "causes_peak_threshold": self.causes_peak_threshold,
            "causes_peak_distance": self.causes_peak_distance,
            "stim_position": [
                signal_1.overload.position,
                signal_2.overload.position,
            ],
        }
        return response, extra

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
        """Counts peaks in the multisensory layer to infer C=1 or C=2."""
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


# =========================================================================
# 3. WIRING: only connects stimuli with the integrator
# =========================================================================

class Wiring:
    """
    Connects a set of Stimulus objects with an Integrator, routing each
    stimulus to the named input that the integrator expects (e.g.
    "auditory", "visual"). It has no architecture or simulation
    parameters of its own: its only responsibility is to know "who is
    who" among the pieces it connects.
    """

    def __init__(self, stimuli, integrator):
        """
        Parameters
        ----------
        stimuli : list of Stimulus
            The two stimuli to connect, in order: the first one goes to
            mode0 and the second one to mode1.
        integrator : object
            Implements `.get_synaps_config()`, `.integrate(...)` and
            `.calculate_causes(...)`.
        """
        self.stimuli_1 = stimuli[0]
        self.stimuli_2 = stimuli[1]
        self.integrator = integrator

    def run(self):
        """Runs the simulation, delegating all the work to the integrator."""

        stimuli_1_duration = (
            self.integrator.time_range[1]
            if self.stimuli_1.duration is None
            else self.stimuli_1.duration
        )
        stimuli_2_duration = (
            self.integrator.time_range[1]
            if self.stimuli_2.duration is None
            else self.stimuli_2.duration
        )

        point_mode0_stimuli = calculate_stimuli_input(
            neurons=self.integrator.neurons,
            intensity=self.stimuli_1.intensity,
            scale=self.stimuli_1.sigma,
            loc=self.stimuli_1.position,
        )
        point_mode1_stimuli = calculate_stimuli_input(
            neurons=self.integrator.neurons,
            intensity=self.stimuli_2.intensity,
            scale=self.stimuli_2.sigma,
            loc=self.stimuli_2.position,
        )

        mode0_stimuli = create_unimodal_stimuli_matrix(
            neurons=self.integrator.neurons,
            stimuli=point_mode0_stimuli,
            stimuli_duration=stimuli_1_duration,
            onset=self.stimuli_1.onset,
            simulation_length=self.integrator.time_range[1],
            time_res=self.integrator.time_res,
            dt=self.integrator.time_res,
            stimuli_n=self.stimuli_1.stim_n,
            soa=self.stimuli_1.soa,
        )
        mode1_stimuli = create_unimodal_stimuli_matrix(
            neurons=self.integrator.neurons,
            stimuli=point_mode1_stimuli,
            stimuli_duration=stimuli_2_duration,
            onset=self.stimuli_2.onset,
            simulation_length=self.integrator.time_range[1],
            time_res=self.integrator.time_res,
            dt=self.integrator.time_res,
            stimuli_n=self.stimuli_2.stim_n,
            soa=self.stimuli_2.soa,
        )

        config = self.integrator.get_synaps_config()

        signal_1 = Signal(
            name=self.stimuli_1.modality,
            overload=self.stimuli_1,
            latsynapses=config["mode0_latsynapses"],
            crossmodal_synapses=config["mode1_to_mode0_synapses"],
            feedforward_synapses=config["mode0_to_multi_synapses"],
            multi_latsynapses=config["multi_latsynapses"],
            unimodal_matrix=mode0_stimuli,
        )
        signal_2 = Signal(
            name=self.stimuli_2.modality,
            overload=self.stimuli_2,
            latsynapses=config["mode1_latsynapses"],
            crossmodal_synapses=config["mode0_to_mode1_synapses"],
            feedforward_synapses=config["mode1_to_multi_synapses"],
            multi_latsynapses=config["multi_latsynapses"],
            unimodal_matrix=mode1_stimuli,
        )

        res, extra = self.integrator.integrate(signal_1, signal_2)

        _res = {
            self.stimuli_1.modality: res["mode0"],
            self.stimuli_2.modality: res["mode1"],
            "multi": res["multi"],
        }

        return _res, extra

    def calculate_causes(self, response, extra):
        """Shortcut to call integrator.calculate_causes(...)."""
        return self.integrator.calculate_causes(multi=response["multi"], **extra)


