import copy
import numpy as np

from ..integrator import Integrator
from neuromsi.utils.readout_tools import calculate_spatiotemporal_causes_from_peaks

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

    def sigmoid(self, u):
        """Sigmoid activation function F(u), shared by the 3 layers."""
        return 1 / (1 + np.exp(-self.s * (u - self.theta)))

    def check_stimuli_compatible(self, modalities):
        """Cuppini2017 requires exactly two modalities: visual and auditory."""
        modalities = list(modalities)
        if len(modalities) != 2:
            raise ValueError(
                "Cuppini2017 expects exactly 2 stimuli (one visual "
                f"and one auditory), but {len(modalities)} were received."
            )
        if set(modalities) != {"visual", "auditory"}:
            raise ValueError(
                "Cuppini2017 expects exactly one visual stimulus and "
                f"one auditory one. Received: {modalities}."
            )

    def get_conf(self):

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
        
    def integrate(self, signal_1, signal_2, time_range, time_res, random):

        hist_times = np.arange(
                            time_range[0], time_range[1], time_res
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
                    signal_1.payload.intensity * self.noise_level
                ) + (
                    2 * signal_1.payload.intensity * self.noise_level
                ) * random.random(self.neurons)
                mode1_noise = -(
                    signal_2.payload.intensity * self.noise_level
                ) + (
                    2 * signal_2.payload.intensity * self.noise_level
                ) * random.random(self.neurons)
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
            mode0_y = mode0_y + time_res * (
                (-mode0_y + self.sigmoid(u_a)) / tau_mode0
            )
            mode1_y = mode1_y + time_res * (
                (-mode1_y + self.sigmoid(u_v)) / tau_mode1
            )
            multi_y = multi_y + time_res * (
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
            "stim_position": [
                signal_1.payload.position,
                signal_2.payload.position,
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

