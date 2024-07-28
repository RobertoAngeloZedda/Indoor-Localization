import numpy as np
from scipy import signal
from Layer import Layer
from Convolutional import Convolutional

class ConvolutionalRGB(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        self.r_layer = Convolutional(input_shape, kernel_size, depth)
        self.g_layer = Convolutional(input_shape, kernel_size, depth)
        self.b_layer = Convolutional(input_shape, kernel_size, depth)

    def forward(self, input):
        if input.ndim == 3:
            input = np.asarray([input])
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

if __name__ == '__main__':

    '''input = []
    c = 0
    for _ in range(5*5*3):
        input.append(c)
        c+=1
    
    input = np.reshape(input, (5,5,3))

    kernel = [[[1, 1, 1], [0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0], [1, 1, 1]]]
    kernel = np.asarray(kernel)

    bias = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    bias = np.asarray(bias)

    print('Shapes:', input.shape, kernel.shape, bias.shape)

    input_by_channels = np.transpose(input, (2, 0, 1))
    
    output = []

    for ch, b, k in zip(input_by_channels, bias, kernel):
        tmp = np.copy(b)
        tmp += signal.correlate2d(ch, k, "valid")
        output.append(tmp)
    
    output = np.asarray(output)
    output = np.transpose(output, (1, 2, 0))

    print(output)
    print(output.shape)'''

    #input = []
    #for _ in range(5*5*3):
    #    input.append(1)

    input = []
    c = 0
    for _ in range(5*5*3):
        input.append(c)
        c+=0.1
    
    input = np.reshape(input, (5,5,3))
    
    input = [input]
    input = np.reshape(input, (1,5,5,3))

    conv = ConvolutionalRGB((1, 5, 5), 3, 1)

    output = conv.forward(input)

    print(output)

    input2 = conv.backward(output, 1)

    print(input2)