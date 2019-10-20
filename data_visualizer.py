import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import json, os
from perceptron import Perceptron
from sigmoidneuron import SigmoidNeuron
from reluneuron import ReLUNeuron
import time


class DataVisualizer:

    def __init__(self, neuron_type):  
        self.covariance_min = -7
        self.covariance_max = 7
        self.locus_min = -10
        self.locus_max = 10
        self.default_num_of_modes = 1
        self.default_num_of_samples = 100
        self.default_json_filename = 'data.txt'
        self.c0_color = 'red'
        self.c1_color = 'blue'
        self.x0 = []
        self.y0 = []
        self.x1 = []
        self.y1 = []
        self.xlim = []
        self.ylim = []

        self.boundary_x = np.array([])
        self.boundary_y = np.array([])
        self.boundary_error_threshold = 1

        self.neuron = neuron_type
        
        self.fig, self.ax = plt.subplots()
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.subplots_adjust(top=0.85, bottom=0.3)

        covariance_min_textbox_ax = plt.axes([0.3, 0.93, 0.05, 0.05])
        self.covariance_min_textbox = TextBox(covariance_min_textbox_ax, 'covar. val. range: ', initial=str(self.covariance_min))
        covariance_max_textbox_ax = plt.axes([0.39, 0.93, 0.05, 0.05])
        self.covariance_max_textbox = TextBox(covariance_max_textbox_ax, 'to ', initial=str(self.covariance_max))

        locus_min_textbox_ax = plt.axes([0.75, 0.93, 0.05, 0.05])
        self.locus_min_textbox = TextBox(locus_min_textbox_ax, 'locus val. range: ', initial=str(self.locus_min))
        locus_max_textbox_ax = plt.axes([0.84, 0.93, 0.05, 0.05])
        self.locus_max_textbox = TextBox(locus_max_textbox_ax, 'to ', initial=str(self.locus_max))

        c0_mode_num_textbox_ax = plt.axes([0.2, 0.12, 0.07, 0.05])
        self.c0_mode_num_textbox = TextBox(c0_mode_num_textbox_ax, 'c0 modes: ', initial=str(self.default_num_of_modes))

        c1_mode_num_textbox_ax = plt.axes([0.2, 0.03, 0.07, 0.05])
        self.c1_mode_num_textbox = TextBox(c1_mode_num_textbox_ax, 'c1 modes: ', initial=str(self.default_num_of_modes))

        c0_sample_num_textbox_ax = plt.axes([0.43, 0.12, 0.1, 0.05])
        self.c0_sample_num_textbox = TextBox(c0_sample_num_textbox_ax, 'c0 samples: ', initial=str(self.default_num_of_samples))

        c1_sample_num_textbox_ax = plt.axes([0.43, 0.03, 0.1, 0.05])
        self.c1_sample_num_textbox = TextBox(c1_sample_num_textbox_ax, 'c1 samples: ', initial=str(self.default_num_of_samples))
       
        generate_button_ax = plt.axes([0.6, 0.12, 0.14, 0.05])
        generate_button = Button(generate_button_ax, 'Generate')
        generate_button.hovercolor = 'lightgreen'
        generate_button.on_clicked(lambda _: self.generate_and_plot_samples(int(self.c0_mode_num_textbox.text), int(self.c0_sample_num_textbox.text), int(self.c1_mode_num_textbox.text), int(self.c1_sample_num_textbox.text)))

        find_boundary_button_ax = plt.axes([0.76, 0.12, 0.19, 0.05])
        find_boundary_button = Button(find_boundary_button_ax, 'Find Boundary')
        find_boundary_button.hovercolor = 'lightgreen'
        find_boundary_button.on_clicked(lambda _: self.find_and_plot_boundary())

        save_textbox_ax = plt.axes([0.6, 0.03, 0.2, 0.05])
        save_textbox = TextBox(save_textbox_ax, '', initial=str(self.default_json_filename))
        
        save_button_ax = plt.axes([0.8, 0.03, 0.15, 0.05])
        save_button = Button(save_button_ax, 'Save to file')
        save_button.hovercolor = 'lightgreen'
        save_button.on_clicked(lambda _: self.save_data_to_json_file(save_textbox.text))
        
        plt.show()
        
    def get_rand_float(self, min, max):
        return min + (max - min) * np.random.random()

    def get_rand_2d_locus(self):
        return self.get_rand_float(int(self.locus_min_textbox.text), int(self.locus_max_textbox.text)), self.get_rand_float(int(self.locus_min_textbox.text), int(self.locus_max_textbox.text))

    def get_rand_2d_covariance(self):
        covariance_min = int(self.covariance_min_textbox.text)
        covariance_max = int(self.covariance_max_textbox.text)
        diagonal_element = self.get_rand_float(covariance_min, covariance_max)
        return [[self.get_rand_float(covariance_min, covariance_max), diagonal_element], 
                [diagonal_element, self.get_rand_float(covariance_min, covariance_max)]]

    def generate_samples(self, c0_num_of_modes, c0_num_of_samples, c1_num_of_modes, c1_num_of_samples):
        self.x0, self.y0 = self.get_2d_multivariate_samples(c0_num_of_modes, c0_num_of_samples)
        self.x1, self.y1 = self.get_2d_multivariate_samples(c1_num_of_modes, c1_num_of_samples)
        self.boundary_x = np.array([])
        self.boundary_y = np.array([])

    def plot_samples(self):
        self.ax.cla()
        self.ax.scatter(self.x0, self.y0, color=self.c0_color, label='class 0')
        self.ax.scatter(self.x1, self.y1, color=self.c1_color, label='class 1')
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_xlabel('X values')
        self.ax.set_ylabel('Y values')
        

    def generate_and_plot_samples(self, c0_num_of_modes, c0_num_of_samples, c1_num_of_modes, c1_num_of_samples):
        self.generate_samples(c0_num_of_modes, c0_num_of_samples, c1_num_of_modes, c1_num_of_samples)
        self.plot_samples()
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()
        plt.show()

    def get_2d_multivariate_samples(self, num_of_modes, num_of_samples):
        x = []
        y = []
        for mode in range(num_of_modes):
            x_temp, y_temp = np.random.multivariate_normal(self.get_rand_2d_locus(), self.get_rand_2d_covariance(), size=num_of_samples, check_valid='ignore').T
            x = np.concatenate((x, x_temp), axis=0)
            y = np.concatenate((y, y_temp), axis=0)
        return x, y

    def get_formatted_data_vector(self):
        data = {}
        data['inputs'] = []
        data['label'] = []
        data['inputs'].append(list(self.x0) + list(self.x1))
        data['inputs'].append(list(self.y0) + list(self.y1))
        for index in range(len(self.x0)):
            data['label'] += [0]
        for index in range(len(self.x1)):
            data['label'] += [1]
        return data
    
    def save_data_to_json_file(self, filename='data.txt'):
        home_dir = os.path.expanduser('~')
        desktop_dir = os.path.join(home_dir, 'OneDrive\\Documents\\ai_labs')
        with open(os.path.join(desktop_dir, filename),'w') as savefile:
            savefile.write(json.dumps(self.get_formatted_data_vector()))

    def load_data_to_neuron(self):
        self.neuron.load_data(self.get_formatted_data_vector())

    def find_boundary(self):
        num_of_iterations = 1000
        i = 0
        start_time = time.process_time()
        while(self.neuron.get_error() >= self.boundary_error_threshold and i < num_of_iterations):
            self.neuron.learning_rate = self.neuron.default_learning_rate * (1 - i/num_of_iterations)
            self.neuron.iterate_weights()
            i += 1
        elapsed_time = time.process_time() - start_time
        self.ax.set_title('error = ' + str(format(self.neuron.get_error(), '.3f')) + ", time = " + str(format((elapsed_time), '.3f')) + ', iterations = ' + str(i) + ', final l_r = ' + str(format(self.neuron.learning_rate, '.3f')))
        
        self.boundary_x = np.linspace(np.amin(self.neuron.inputs[1]), np.amax(self.neuron.inputs[1]))
        self.boundary_y = -(self.neuron.weights[0] + self.neuron.weights[1] * self.boundary_x)/self.neuron.weights[2] 
        
        
    def plot_boundary(self):
        boundary_color = 'magenta'
        boundary_alpha = 0.6
        self.ax.plot(self.boundary_x, self.boundary_y, color=boundary_color, alpha=boundary_alpha)

    def plot_half_planes(self):
        half_plane_alpha = 0.15

        boundary_min = np.amin([np.amin(self.neuron.inputs[2]), np.amin(self.boundary_y)])
        boundary_max = np.amax([np.amax(self.neuron.inputs[2]), np.amax(self.boundary_y)])
        
        under_the_border = self.neuron.get_state([self.neuron.inputs[0][0], self.neuron.inputs[1][0], boundary_min - 1])
        if self.neuron.activate([under_the_border]) <= 0.5:
            self.ax.fill_between(self.boundary_x, self.boundary_y, boundary_min, color=self.c0_color, alpha=half_plane_alpha)
            self.ax.fill_between(self.boundary_x, self.boundary_y, boundary_max, color=self.c1_color, alpha=half_plane_alpha)
        else:
            self.ax.fill_between(self.boundary_x, self.boundary_y, boundary_min, color=self.c1_color, alpha=half_plane_alpha)
            self.ax.fill_between(self.boundary_x, self.boundary_y, boundary_max, color=self.c0_color, alpha=half_plane_alpha)
        

    def find_and_plot_boundary(self, neuron_type=Perceptron()):
        # self.neuron = neuron_type
        self.load_data_to_neuron()
        self.find_boundary()
        self.plot_boundary()
        self.plot_half_planes()
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        plt.show()
    
dv = DataVisualizer(SigmoidNeuron())
