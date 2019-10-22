import numpy as np
from neuron import Neuron

class SinusoidNeuron(Neuron):
    def activate(self, state):
        return np.sin(state)

    def activate_prime(self, state):
        return np.cos(state)