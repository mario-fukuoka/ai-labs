import numpy as np
from neuron import Neuron

class Perceptron(Neuron):
    def activate(self, state):
        return np.heaviside(state, 1)

    def activate_prime(self, state):
        return np.ones(len(state))