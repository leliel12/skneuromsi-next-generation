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

    @classmethod
    def get_stimuli(cls):
        """Devuelve una instancia de la clase con valores por defecto."""
        return cls(
            position=90,
            intensity=27,
            sigma=4,
            onset=0,
            duration=100,
        )


class Visual(Stimulus):
    """Un estimulo visual (p.ej. un flash de luz en una posicion)."""

    modality = "visual"

    @classmethod
    def get_stimuli(cls):
        """Devuelve un estimulo visual con los valores por defecto."""
        return cls(
            position=90,
            intensity=27,
            sigma=4,
            onset=0,
            duration=100,
        )


class Audio(Stimulus):
    """Un estimulo auditivo (p.ej. un beep en una posicion)."""

    modality = "auditory"

    @classmethod
    def get_stimuli(cls):
        """Devuelve un estimulo auditivo con los valores por defecto."""
        return cls(
            position=100,
            intensity=28,
            sigma=32,
            onset=0,
            duration=100,
        )