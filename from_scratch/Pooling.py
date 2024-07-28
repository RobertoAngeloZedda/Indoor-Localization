import numpy as np
from Layer import Layer

class Pooling_max(Layer):
    def __init__(self, input_shape, kernel_shape, stride=(1, 1)):
        self.depth, self.input_height, self.input_widht = input_shape
        self.kernel_height, self.kernel_widht = kernel_shape
        self.stride_height, self.stride_widht = stride

        self.output_height = int((self.input_height - self.kernel_height) / self.stride_height + 1)
        self.output_widht = int((self.input_widht - self.kernel_widht) / self.stride_widht + 1)

    def forward(self, input):
        self.input = input

        output = np.zeros((self.depth, self.output_height, self.output_widht))
        for d in range(self.depth):
            for i in range(self.output_height):
                for j in range(self.output_widht):
                    start_i = i * self.stride_height
                    end_i = start_i + self.kernel_height
                    start_j = j * self.stride_widht
                    end_j = start_j + self.kernel_widht

                    output[d, i, j] = np.max(input[d, start_i:end_i, start_j:end_j])
        
        return output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros((self.depth, self.input_height, self.input_widht))
        for d in range(self.depth):
            for i in range(self.output_height):
                for j in range(self.output_widht):
                    start_i = i * self.stride_height
                    end_i = start_i + self.kernel_height
                    start_j = j * self.stride_widht
                    end_j = start_j + self.kernel_widht

                    old_sub_matrix = self.input[d, start_i:end_i, start_j:end_j]

                    mask = (old_sub_matrix == np.max(old_sub_matrix))

                    input_gradient[d, start_i:end_i, start_j:end_j] += mask * output_gradient[d, i, j]

        return input_gradient

if __name__ == '__main__':
    example = np.random.randint(low=0, high=10, size=(2, 6, 6))
    print(example)

    p = Pooling_max((2, 6, 6), (2, 2))
    
    out = p.forward(example)
    print(out)

    grad = np.ones_like(out)
    out = p.backward(out)
    print(out)