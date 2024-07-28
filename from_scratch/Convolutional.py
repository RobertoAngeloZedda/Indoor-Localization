import numpy as np
from scipy import signal
from Layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.depth = depth

        self.output_height = self.input_height - kernel_size + 1
        self.output_width = self.input_width - kernel_size + 1

        self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        
        self.biases = np.random.randn(depth, self.output_height, self.output_width)
        

    def forward(self, input):
        self.input = input

        output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros((self.input_depth, self.input_height, self.input_width))

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient