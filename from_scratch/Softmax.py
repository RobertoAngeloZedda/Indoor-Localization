import numpy as np
from Layer import Layer

class Softmax(Layer):
    def forward(self, input):
        exp = np.exp(input)
        self.output = exp / np.sum(exp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        tmp = np.tile(self.output, n)
        return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)