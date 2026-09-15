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

@dataclass
class Signal:
    """
    Empaqueta UN estimulo (overload) junto con todos los valores que le
    corresponden a esa modalidad: duracion, matriz temporal, input puntual,
    sinapsis laterales de su capa, sinapsis cross-modales que recibe desde
    la otra capa, y sinapsis feedforward hacia la capa multisensorial.

    Tambien lleva copiados los valores compartidos de la simulacion
    (n_time_steps,
    hist_times, multi_latsynapses) para que cada Signal sea
    autocontenido. Wiring construye uno por estimulo y se los pasa a
    integrate().
    """

    name: str ## es necesario?
    overload: Stimulus # idem, enrealidad termina siendo innecesaria tanta info
    
    unimodal_matrix: np.ndarray
    latsynapses: np.ndarray
    crossmodal_synapses: np.ndarray
    feedforward_synapses: np.ndarray
    multi_latsynapses: np.ndarray ## raro
    
    
# =========================================================================
# 2. INTEGRADOR: base generica + arquitectura Cuppini (2017)
# =========================================================================

class Integrator:
    """
    Clase base de cualquier arquitectura neuronal.

    Contiene lo que es comun a todas: los metodos para calcular la
    conectividad sinaptica (lateral e inter-areal). Cada arquitectura
    concreta (p.ej. Cuppini2017) hereda de aca y agrega sus propios
    parametros, su dinamica y su mecanismo de inferencia causal.
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
            sigma=self.cross_modal_sigma,
        )
        mode_2_to_mode_1_synapses = self.calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.cross_modal_weight,
            sigma=self.cross_modal_sigma,
        )

        # --- Sinapsis feedforward hacia la capa multisensorial (Wma, Wmv) ---
        mode_1_to_multi_synapses = self.calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.feedforward_weight,
            sigma=self.feedforward_sigma,
        )
        mode_2_to_multi_synapses = self.calculate_inter_areal_synapses(
            neurons=self.neurons,
            weight=self.feedforward_weight,
            sigma=self.feedforward_sigma,
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

    def __init__(self, stimuli, integrator):
        """
        Parameters
        ----------
        stimuli : list of Stimulus
            Los dos estimulos a conectar, en orden: el primero va a
            mode_1 y el segundo a mode_2.
        integrator : object
            Implementa `.get_synaps_config()`, `.integrate(...)` y
            `.calculate_causes(...)`.
        """
        self.stimuli_1 = stimuli[0]
        self.stimuli_2 = stimuli[1]
        self.integrator = integrator

    def run(self):
        """Corre la simulacion, delegando todo el trabajo al integrador."""       
        
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
        
        point_mode_1_stimuli = calculate_stimuli_input(
            neurons=self.integrator.neurons,
            intensity=self.stimuli_1.intensity,
            scale=self.stimuli_1.sigma,
            loc=self.stimuli_1.position,
        )
        point_mode_2_stimuli = calculate_stimuli_input(
            neurons=self.integrator.neurons,
            intensity=self.stimuli_2.intensity,
            scale=self.stimuli_2.sigma,
            loc=self.stimuli_2.position,
            )
        
        mode_1_stimuli = create_unimodal_stimuli_matrix(
            neurons=self.integrator.neurons,
            stimuli=point_mode_1_stimuli,
            stimuli_duration=stimuli_1_duration,
            onset=self.stimuli_1.onset,
            simulation_length=self.integrator.time_range[1],
            time_res=self.integrator.time_res,
            dt=self.integrator.time_res,
            stimuli_n=self.stimuli_1.stim_n,
            soa=self.stimuli_1.soa,
        )
        mode_2_stimuli = create_unimodal_stimuli_matrix(
            neurons=self.integrator.neurons,
            stimuli=point_mode_2_stimuli,
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
            latsynapses=config["mode_1_latsynapses"],
            crossmodal_synapses=config["mode_2_to_mode_1_synapses"],
            feedforward_synapses=config["mode_1_to_multi_synapses"],
            multi_latsynapses=config["multi_latsynapses"],
            unimodal_matrix=mode_1_stimuli,
        )
        signal_2 = Signal(
            name=self.stimuli_2.modality,
            overload=self.stimuli_2,            
            latsynapses=config["mode_2_latsynapses"],
            crossmodal_synapses=config["mode_1_to_mode_2_synapses"],
            feedforward_synapses=config["mode_2_to_multi_synapses"],
            multi_latsynapses=config["multi_latsynapses"],
            unimodal_matrix=mode_2_stimuli,
        )
        
        res, extra = self.integrator.integrate(signal_1, signal_2)

        _res = {
                self.stimuli_1.modality: res["mode_1"],
                self.stimuli_2.modality: res["mode_2"],
                "multi": res["multi"],
                }

        return _res, extra

    def calculate_causes(self, response, extra):
        """Atajo para llamar a integrador.calculate_causes(...)."""
        return self.integrator.calculate_causes(multi=response["multi"], **extra)


