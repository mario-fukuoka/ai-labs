import numpy as np
from neuron import Neuron

class HyperTanNeuron(Neuron):
    def activate(self, state):
        return np.tanh(state)

    def activate_prime(self, state):
        return np.square(2/(np.exp(state) + np.exp(-state)))
        # sech x = 2/(ex + e-x)