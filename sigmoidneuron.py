import numpy as np
from neuron import Neuron

class SigmoidNeuron(Neuron):
    def activate(self, state):
        beta = -1
        return 1/(1 + np.exp((beta * state)))

    def activate_prime(self, state):
        f = self.activate(state)
        return f * (1 - f)