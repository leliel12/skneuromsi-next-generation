"""
Refactor del modelo de Cuppini et al. (2017) en tres piezas separadas:

1. Stimulus (y sus subclases Visual, Audio):
   Representa UNICAMENTE el evento sensorial externo (posicion, intensidad,
   incertidumbre, timing). No sabe nada de neuronas, arreglos, sigmoides
   ni sinapsis. Es intercambiable entre cualquier integrador futuro.

2. Cuppini2017 (el integrador):
   Contiene TODO lo que es especifico de esta arquitectura en particular:
   cantidad de neuronas, dinamica de la sigmoide, conectividad lateral
   (Mexican Hat), sinapsis cross-modales y feedforward, y el mecanismo
   de inferencia causal (conteo de picos).

3. Wiring:
   Solo conecta estimulos con el integrador, enrutando cada estimulo a
   la entrada correspondiente segun su `.modality`. No tiene parametros
   propios de arquitectura ni de simulacion.

NOTA: las funciones calculate_lateral_synapses, calculate_inter_areal_synapses,
calculate_stimuli_input, create_unimodal_stimuli_matrix y
calculate_spatiotemporal_causes_from_peaks se asumen ya existentes en el
paquete original (se usaban en el codigo monolitico original sin mostrar
su implementacion). Ajustar el import segun donde vivan realmente.
"""

import copy
import numpy as np

from dataclasses import dataclass



from utils.neural_tools import (
    calculate_inter_areal_synapses,
    calculate_lateral_synapses,
    calculate_stimuli_input,
    create_unimodal_stimuli_matrix,
)
from utils.readout_tools import calculate_spatiotemporal_causes_from_peaks


# =========================================================================
# 1. STIMULI: representan SOLO el evento fisico externo
# =========================================================================

class Stimulus:
    """
    Representa un unico estimulo sensorial externo, independiente de
    cualquier arquitectura neuronal que despues lo vaya a procesar.

    Solo describe "que paso en el mundo": donde, que tan fuerte, que tan
    incierto, y cuando. No sabe nada de neuronas, campos receptivos ni
    dinamica de integracion.
    """

    #: Nombre de la modalidad. Wiring lo usa para enrutar este estimulo
    #: a la entrada correspondiente del integrador (p.ej. "visual",
    #: "auditory"). Las subclases lo sobreescriben.
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
            Posicion espacial del estimulo, en grados.
        intensity : float
            Amplitud maxima del estimulo (E0 en la notacion del paper).
        sigma : float
            Dispersion/incertidumbre espacial del estimulo (sigma_c en
            el paper). Describe al estimulo en si, no a un campo
            receptivo neuronal.
        onset : float, optional
            Instante (ms) en que empieza el estimulo. Default 0.
        duration : float o None, optional
            Cuanto dura el estimulo (ms). Si es None, el integrador lo
            interpreta como "dura toda la simulacion".
        stim_n : int, optional
            Cantidad de repeticiones de este estimulo en el tiempo
            (p.ej. dos beeps en vez de uno). Default 1.
        soa : float o None, optional
            Stimulus Onset Asynchrony: separacion temporal (ms) entre
            repeticiones, solo se usa si stim_n > 1.
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
    """Un estimulo visual (p.ej. un flash de luz en una posicion)."""

    modality = "visual"


class Audio(Stimulus):
    """Un estimulo auditivo (p.ej. un beep en una posicion)."""

    modality = "auditory"


# =========================================================================
# 2. INTEGRADOR: todo lo especifico de la arquitectura Cuppini (2017)
# =========================================================================

class Cuppini2017:
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
        tau=(3, 15, 1),            # (auditiva, visual, multisensorial)
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
                "(auditiva, visual, multisensorial)"
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
            "auditory": dict(
                excitation_loc=5, inhibition_loc=4,
                excitation_scale=3, inhibition_scale=120,
            ),
            "visual": dict(
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

    def run(self, stimuli):
        """
        Corre la simulacion dado un diccionario de estimulos, indexado
        por nombre de modalidad.

        Parameters
        ----------
        stimuli : dict
            Debe contener las claves "auditory" y "visual", cada una
            con una instancia de Stimulus.

        Returns
        -------
        response : dict
            Con claves "auditory", "visual", "multi", cada una un array
            de forma (n_time_steps, neurons) con la actividad a lo
            largo del tiempo.
        extra : dict
            Metadata necesaria despues para calculate_causes().
        """
        auditory_stim = stimuli["auditory"]
        visual_stim = stimuli["visual"]

        auditory_duration = (
            self.time_range[1]
            if auditory_stim.duration is None
            else auditory_stim.duration
        )
        visual_duration = (
            self.time_range[1]
            if visual_stim.duration is None
            else visual_stim.duration
        )

        hist_times = np.arange(
            self.time_range[0], self.time_range[1], self.time_res
        )
        n_time_steps = hist_times.size

        # --- Conectividad lateral (Mexican Hat), dentro de cada capa ---
        auditory_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["auditory"]
        )
        visual_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["visual"]
        )
        multi_latsynapses = calculate_lateral_synapses(
            neurons=self.neurons, **self.lateral_params["multi"]
        )

        # --- Sinapsis cross-modales (Wav, Wva) ---
        auditory_to_visual_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.cross_modal_weight,
            sigma=self.cross_modal_sigma,
        )
        visual_to_auditory_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.cross_modal_weight,
            sigma=self.cross_modal_sigma,
        )

        # --- Sinapsis feedforward hacia la capa multisensorial (Wma, Wmv) ---
        auditory_to_multi_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.feedforward_weight,
            sigma=self.feedforward_sigma,
        )
        visual_to_multi_synapses = calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.feedforward_weight,
            sigma=self.feedforward_sigma,
        )

        # --- Construir el input externo (matriz tiempo x neuronas) ---
        # a partir de los datos que vinieron en los objetos Stimulus.
        point_auditory_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=auditory_stim.intensity,
            scale=auditory_stim.sigma,
            loc=auditory_stim.position,
        )
        point_visual_stimuli = calculate_stimuli_input(
            neurons=self.neurons,
            intensity=visual_stim.intensity,
            scale=visual_stim.sigma,
            loc=visual_stim.position,
        )

        auditory_stimuli = create_unimodal_stimuli_matrix(
            neurons=self.neurons,
            stimuli=point_auditory_stimuli,
            stimuli_duration=auditory_duration,
            onset=auditory_stim.onset,
            simulation_length=self.time_range[1],
            time_res=self.time_res,
            dt=self.time_res,
            stimuli_n=auditory_stim.stim_n,
            soa=auditory_stim.soa,
        )
        visual_stimuli = create_unimodal_stimuli_matrix(
            neurons=self.neurons,
            stimuli=point_visual_stimuli,
            stimuli_duration=visual_duration,
            onset=visual_stim.onset,
            simulation_length=self.time_range[1],
            time_res=self.time_res,
            dt=self.time_res,
            stimuli_n=visual_stim.stim_n,
            soa=visual_stim.soa,
        )

        # --- Contenedores de estado ---
        z_1d = np.zeros(self.neurons)
        auditory_y = copy.deepcopy(z_1d)
        visual_y = copy.deepcopy(z_1d)
        multi_y = copy.deepcopy(z_1d)

        z_2d = np.zeros((n_time_steps, self.neurons))
        auditory_res = copy.deepcopy(z_2d)
        visual_res = copy.deepcopy(z_2d)
        multi_res = copy.deepcopy(z_2d)
        del z_1d, z_2d

        tau_a, tau_v, tau_m = self.tau

        for i in range(n_time_steps):
            # Input cruzado (via Wva, Wav)
            auditory_cm_input = np.sum(
                visual_to_auditory_synapses * visual_y, axis=1
            )
            visual_cm_input = np.sum(
                auditory_to_visual_synapses * auditory_y, axis=1
            )

            # Input feedforward hacia la capa multisensorial
            multi_input = (
                np.sum(auditory_to_multi_synapses * auditory_y, axis=1)
                + np.sum(visual_to_multi_synapses * visual_y, axis=1)
            )

            # Input externo (estimulo) + cruzado
            auditory_input = auditory_stimuli[i] + auditory_cm_input
            visual_input = visual_stimuli[i] + visual_cm_input

            # Ruido, si esta activado
            if self.noise:
                auditory_noise = -(
                    auditory_stim.intensity * self.noise_level
                ) + (
                    2 * auditory_stim.intensity * self.noise_level
                ) * self.random.random(self.neurons)
                visual_noise = -(
                    visual_stim.intensity * self.noise_level
                ) + (
                    2 * visual_stim.intensity * self.noise_level
                ) * self.random.random(self.neurons)
                auditory_input = auditory_input + auditory_noise
                visual_input = visual_input + visual_noise

            # Input lateral (Mexican Hat), dentro de cada capa
            la = np.sum(auditory_latsynapses * auditory_y, axis=1)
            lv = np.sum(visual_latsynapses * visual_y, axis=1)
            lm = np.sum(multi_latsynapses * multi_y, axis=1)

            # Input total por capa
            u_a = la + auditory_input
            u_v = lv + visual_input
            u_m = lm + multi_input

            # Paso de Euler: y_nuevo = y + dt * (1/tau) * (-y + sigmoide(u))
            auditory_y = auditory_y + self.time_res * (
                (-auditory_y + self.sigmoid(u_a)) / tau_a
            )
            visual_y = visual_y + self.time_res * (
                (-visual_y + self.sigmoid(u_v)) / tau_v
            )
            multi_y = multi_y + self.time_res * (
                (-multi_y + self.sigmoid(u_m)) / tau_m
            )

            auditory_res[i, :] = auditory_y
            visual_res[i, :] = visual_y
            multi_res[i, :] = multi_y

        response = {
            "auditory": auditory_res,
            "visual": visual_res,
            "multi": multi_res,
        }
        extra = {
            "causes_kind": self.causes_kind,
            "causes_dim": self.causes_dim,
            "causes_peak_threshold": self.causes_peak_threshold,
            "causes_peak_distance": self.causes_peak_distance,
            "stim_position": [auditory_stim.position, visual_stim.position],
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


# =========================================================================
# 3. WIRING: solo conecta estimulos con el integrador
# =========================================================================

class Wiring:
    """
    Conecta un conjunto de objetos Stimulus con un Integrador, enrutando
    cada estimulo a la entrada nombrada que el integrador espera
    (p.ej. "auditory", "visual"). No tiene parametros propios de
    arquitectura ni de simulacion: su unica responsabilidad es saber
    "quien es quien" entre las piezas que conecta.
    """

    def __init__(self, estimulos, integrador):
        """
        Parameters
        ----------
        estimulos : list of Stimulus
            Los estimulos a conectar. Cada uno debe tener un atributo
            `.modality` (p.ej. "visual", "auditory") que se usa como
            clave para `integrador.run(...)`.
        integrador : object
            Debe implementar un metodo `.run(stimuli_dict)`.
        """
        self.estimulos = {stim.modality: stim for stim in estimulos}
        self.integrador = integrador

    def run(self):
        """Corre la simulacion, delegando todo el trabajo al integrador."""
        return self.integrador.run(self.estimulos)

    def calculate_causes(self, response, extra):
        """Atajo para llamar a integrador.calculate_causes(...)."""
        return self.integrador.calculate_causes(multi=response["multi"], **extra)


# =========================================================================
# EJEMPLO DE USO (replica la sintaxis de la imagen)
# =========================================================================

if __name__ == "__main__":
    ojos = Visual(position=90, intensity=27, sigma=4, onset=0, duration=100)
    oreja = Audio(position=100, intensity=28, sigma=32, onset=0, duration=100)

    cerebro = Cuppini2017(
        neurons=180,
        tau=(3, 15, 1),
        s=0.3,
        theta=20,
        cross_modal_weight=1.4,
        feedforward_weight=18,
        noise=True,
        noise_level=0.40,
        seed=42,
        time_range=(0, 100),
        time_res=0.01,
    )

    model = Wiring(estimulos=[ojos, oreja], integrador=cerebro)
    response, extra = model.run()

    causes = model.calculate_causes(response, extra)
    print(causes)