import copy
import numpy as np

from .integrator import Integrator
from utils.readout_tools import calculate_spatiotemporal_causes_from_peaks


class Cuppini2017(Integrator):
    """
    Integrador neuronal que implementa la arquitectura de
    Cuppini et al. (2017): dos capas unisensoriales (auditiva, visual)
    reciprocamente conectadas, que alimentan una tercera capa
    multisensorial ("area de inferencia causal").

    Cada parametro aca es especifico de ESTA arquitectura: cantidad de
    neuronas, dinamica de la sigmoide, conectividad lateral (Mexican Hat),
    y fuerza de las sinapsis cross-modales / feedforward. Un integrador
    distinto podria usar parametros completamente diferentes, o un
    mecanismo distinto, y aun asi aceptar los mismos objetos Stimulus.
    """

    def __init__(
        self,
        *,
        neurons=180,
        tau=(3, 15, 1),            # (mode_1, mode_2, multisensorial)
        s=0.3,
        theta=20,
        cross_modal_weight=1.4,     # W0 de Wav / Wva
        cross_modal_sigma=5,
        feedforward_weight=18,      # W0 de Wma / Wmv
        feedforward_sigma=0.5,
        lateral_params=None,        # Mexican Hat, uno por capa
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
                "tau debe tener exactamente 3 valores: "
                "(mode_1, mode_2, multisensorial)"
            )

        self.neurons = neurons
        self.tau = tau
        self.s = s
        self.theta = theta

        self.cross_modal_weight = cross_modal_weight
        self.cross_modal_sigma = cross_modal_sigma
        self.feedforward_weight = feedforward_weight
        self.feedforward_sigma = feedforward_sigma

        # Valores por defecto del Mexican Hat, uno por capa, iguales a
        # los que estaban hardcodeados en el run() monolitico original.
        self.lateral_params = lateral_params or {
            "mode_1": dict(
                excitation_loc=5, inhibition_loc=4,
                excitation_scale=3, inhibition_scale=120,
            ),
            "mode_2": dict(
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
        """Funcion de activacion sigmoide F(u), compartida por las 3 capas."""
        return 1 / (1 + np.exp(-self.s * (u - self.theta)))

    def get_synaps_config(self):

        # --- Conectividad lateral (Mexican Hat), dentro de cada capa ---
        mode_1_latsynapses = self.calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["mode_1"]
        )
        mode_2_latsynapses = self.calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["mode_2"]
        )
        multi_latsynapses = self.calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["multi"]
        )

        # --- Sinapsis cross-modales (Wav, Wva) ---
        mode_1_to_mode_2_synapses = self.calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.cross_modal_weight,
            sigma=5,
        )
        mode_2_to_mode_1_synapses = self.calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.cross_modal_weight,
            sigma=5,
        )

        # --- Sinapsis feedforward hacia la capa multisensorial (Wma, Wmv) ---
        mode_1_to_multi_synapses = self.calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.feedforward_weight,
            sigma=0.5,
        )
        mode_2_to_multi_synapses = self.calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.feedforward_weight,
            sigma=0.5,
        )

        return {
            "mode_1_latsynapses": mode_1_latsynapses,
            "mode_2_latsynapses": mode_2_latsynapses,
            "multi_latsynapses": multi_latsynapses,
            "mode_1_to_mode_2_synapses": mode_1_to_mode_2_synapses,
            "mode_2_to_mode_1_synapses": mode_2_to_mode_1_synapses,
            "mode_1_to_multi_synapses": mode_1_to_multi_synapses,
            "mode_2_to_multi_synapses": mode_2_to_multi_synapses,
        }

    def integrate(self, signal_1, signal_2):

        hist_times = np.arange(
                            self.time_range[0], self.time_range[1], self.time_res
                        )
        n_time_steps = hist_times.size

        # --- Contenedores de estado ---
        z_1d = np.zeros(self.neurons)
        mode_1_y = copy.deepcopy(z_1d)
        mode_2_y = copy.deepcopy(z_1d)
        multi_y = copy.deepcopy(z_1d)


        z_2d = np.zeros((n_time_steps, self.neurons))
        mode_1_res = copy.deepcopy(z_2d)
        mode_2_res = copy.deepcopy(z_2d)
        multi_res = copy.deepcopy(z_2d)
        del z_1d, z_2d

        tau_mode_1, tau_mode_2, tau_multi = self.tau

        for i in range(n_time_steps):
            # Input cruzado (via Wva, Wav)
            mode_1_cm_input = np.sum(
                signal_1.crossmodal_synapses * mode_2_y, axis=1
            )
            mode_2_cm_input = np.sum(
                signal_2.crossmodal_synapses * mode_1_y, axis=1
            )

            # Input feedforward hacia la capa multisensorial
            multi_input = (
                np.sum(signal_1.feedforward_synapses * mode_1_y, axis=1)
                + np.sum(signal_2.feedforward_synapses * mode_2_y, axis=1)
            )

            # Input externo (estimulo) + cruzado
            mode_1_input = signal_1.unimodal_matrix[i] + mode_1_cm_input
            mode_2_input = signal_2.unimodal_matrix[i] + mode_2_cm_input

            # Ruido, si esta activado
            if self.noise:
                mode_1_noise = -(
                    signal_1.overload.intensity * self.noise_level
                ) + (
                    2 * signal_1.overload.intensity * self.noise_level
                ) * self.random.random(self.neurons)
                mode_2_noise = -(
                    signal_2.overload.intensity * self.noise_level
                ) + (
                    2 * signal_2.overload.intensity * self.noise_level
                ) * self.random.random(self.neurons)
                mode_1_input = mode_1_input + mode_1_noise
                mode_2_input = mode_2_input + mode_2_noise

            # Input lateral (Mexican Hat), dentro de cada capa
            la = np.sum(signal_1.latsynapses * mode_1_y, axis=1)
            lv = np.sum(signal_2.latsynapses * mode_2_y, axis=1)
            lm = np.sum(signal_1.multi_latsynapses * multi_y, axis=1)

            # Input total por capa
            u_a = la + mode_1_input
            u_v = lv + mode_2_input
            u_m = lm + multi_input

            # Paso de Euler: y_nuevo = y + dt * (1/tau) * (-y + sigmoide(u))
            mode_1_y = mode_1_y + self.time_res * (
                (-mode_1_y + self.sigmoid(u_a)) / tau_mode_1
            )
            mode_2_y = mode_2_y + self.time_res * (
                (-mode_2_y + self.sigmoid(u_v)) / tau_mode_2
            )
            multi_y = multi_y + self.time_res * (
                (-multi_y + self.sigmoid(u_m)) / tau_multi
            )

            mode_1_res[i, :] = mode_1_y
            mode_2_res[i, :] = mode_2_y
            multi_res[i, :] = multi_y

        response = {
            "mode_1": mode_1_res,
            "mode_2": mode_2_res,
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
        """Cuenta picos en la capa multisensorial para inferir C=1 o C=2."""
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