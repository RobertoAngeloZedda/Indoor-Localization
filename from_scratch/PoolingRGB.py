import numpy as np
from Layer import Layer
from Pooling import Pooling_max

class Pooling_maxRGB(Layer):
    def __init__(self, input_shape, kernel_shape, stride=(1, 1)):
        self.r_layer = Pooling_max(input_shape, kernel_shape, stride)
        self.g_layer = Pooling_max(input_shape, kernel_shape, stride)
        self.b_layer = Pooling_max(input_shape, kernel_shape, stride)
    
    def forward(self, input):
        input = np.transpose(input, (3, 0, 1, 2))

        r_out = self.r_layer.forward(input[0])
        g_out = self.r_layer.forward(input[1])
        b_out = self.r_layer.forward(input[2])

        output = [r_out, g_out, b_out]
        output = np.asarray(output)
        output = np.transpose(output, (1, 2, 3, 0))
        return output

    def backward(self, output_gradient, learning_rate):
        output_gradient = np.transpose(output_gradient, (3, 0, 1, 2))

        r_in_grad = self.r_layer.backward(output_gradient[0], learning_rate)
        g_in_grad = self.r_layer.backward(output_gradient[1], learning_rate)
        b_in_grad = self.r_layer.backward(output_gradient[2], learning_rate)

        input_gradient = [r_in_grad, g_in_grad, b_in_grad]
        input_gradient = np.asarray(input_gradient)
        input_gradient = np.transpose(input_gradient, (1, 2, 3, 0))
        return input_gradient