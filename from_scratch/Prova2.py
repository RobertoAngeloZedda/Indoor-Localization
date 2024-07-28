import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from Convolutional import Convolutional
from Activation import Activation
from Functions import sigmoid, sigmoid_derivative, mse, mse_derivative
from Dense import Dense
from Reshape import Reshape
from Network import Network

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    two_index = np.where(y == 2)[0][:limit]

    all_indices = np.hstack((zero_index, one_index, two_index))
    all_indices = np.random.permutation(all_indices)
    
    x, y = x[all_indices], y[all_indices]

    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255

    y = to_categorical(y, num_classes = 3)
    y = y.reshape(len(y), 3, 1)
    
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = Network([
    Convolutional((1, 28, 28), 3, 7),
    Activation(sigmoid, sigmoid_derivative),
    Convolutional((7, 26, 26), 5, 5),
    Activation(sigmoid, sigmoid_derivative),
    Reshape((5, 22, 22), (5 * 22 * 22, 1)),
    Dense(5 * 22 * 22, 100),
    Activation(sigmoid, sigmoid_derivative),
    Dense(100, 3),
    Activation(sigmoid, sigmoid_derivative),
])

# train
network.train(loss=mse,
              loss_derivative=mse_derivative,
              X=x_train, 
              Y=y_train, 
              epochs=20,
              learning_rate=0.05, 
              print_debug=True)

# test
count = 0
for x, y in zip(x_test, y_test):
    output = network.predict(x)
    print(f"pred: {np.argmax(output)} ({np.max(output)}), \ttrue: {np.argmax(y)}")
    
    if np.argmax(output) == np.argmax(y):
        count += 1

print(f'{count}/{len(x_test)}')