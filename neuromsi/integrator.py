import numpy as np

from utils.neural_tools import calculate_neural_distance


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