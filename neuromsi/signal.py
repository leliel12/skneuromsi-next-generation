import numpy as np

from dataclasses import dataclass

from .stimulus import Stimulus


@dataclass
class Signal:
    """
    Empaqueta UN estimulo (overload) junto con todos los valores que le
    corresponden a esa modalidad: duracion, matriz temporal, input puntual,
    sinapsis laterales de su capa, sinapsis cross-modales que recibe desde
    la otra capa, y sinapsis feedforward hacia la capa multisensorial.

    Tambien lleva copiados los valores compartidos de la simulacion
    (multi_latsynapses) para que cada Signal sea autocontenido. Wiring
    construye uno por estimulo y se los pasa a integrate().
    """

    name: str ## es necesario?
    overload: Stimulus # idem, enrealidad termina siendo innecesaria tanta info

    unimodal_matrix: np.ndarray
    latsynapses: np.ndarray
    crossmodal_synapses: np.ndarray
    feedforward_synapses: np.ndarray
    multi_latsynapses: np.ndarray ## raro