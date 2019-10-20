import numpy as np
from neuron import Neuron

class ReLUNeuron(Neuron):
    def activate(self, state):
        return np.maximum(np.zeros(len(state)), state)

    def activate_prime(self, state):
        derivatives = []
        for s in state:
            if s > 0:
                derivatives.append(1)
            else:
                derivatives.append(0)
        return np.array(derivatives)