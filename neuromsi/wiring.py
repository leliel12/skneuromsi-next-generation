from .signal import Signal
from utils.neural_tools import (
    calculate_stimuli_input,
    create_unimodal_stimuli_matrix,
)


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