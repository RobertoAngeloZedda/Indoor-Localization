import numpy as np
from Layer import Layer

class Activation(Layer):
    def __init__(self, function, function_derivative):
        self.function = function
        self.function_derivative = function_derivative

    def forward(self, input):
        self.input = input

        return self.function(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.function_derivative(self.input))