import numpy as np
from abc import ABC, abstractmethod

class Neuron(ABC):
    def __init__(self):
        self.num_of_inputs = 0
        self.weights = np.array([])
        self.default_learning_rate = 0.5
        self.learning_rate = self.default_learning_rate
        self.change_in_weights = np.array([])
        self.inputs = np.array([])
        self.actual_label = np.array([])
        self.predicted_label = np.array([])

    def load_data(self, data):
        self.learning_rate = self.default_learning_rate
        self.num_of_inputs = len(data['inputs']) + 1
        self.weights = np.ones(self.num_of_inputs)
        self.change_in_weights = np.zeros(self.num_of_inputs)
        self.actual_label = data['label']
        self.inputs = np.insert(data['inputs'], 0, np.ones(len(self.actual_label)), axis=0)
        self.predicted_label = self.activate(self.get_state(self.inputs))

    def iterate_weights(self):
        state = self.get_state(self.inputs)
        activation_prime = self.activate_prime(state)
        for input_index in range(self.num_of_inputs):
            self.change_in_weights[input_index] = self.learning_rate * np.sum((self.actual_label - self.predicted_label) * activation_prime * self.inputs[input_index])
        self.weights += self.change_in_weights
        self.predicted_label = self.activate(state)

    def get_error(self):
        return np.sum(np.abs(self.actual_label - self.predicted_label))

    def get_state(self, inputs):
        return np.transpose(self.weights) @ inputs

    @abstractmethod
    def activate(self, state):
        pass
        # define activation on a state
        # should return np.array() of labels/activation values

    @abstractmethod
    def activate_prime(self, state):
        pass
        # define the derivative of the activation function
        # should return np.array() of the derivative values per each state
