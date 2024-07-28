import numpy as np
from Dense import Dense
from Activation import Activation
from Network import Network
from Functions import tanh, tanh_derivative, mse, mse_derivative

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = Network([
    Dense(2, 5),
    Activation(np.tanh, tanh_derivative),
    Dense(5, 4),
    Activation(np.tanh, tanh_derivative),
    Dense(4, 1),
    Activation(tanh, tanh_derivative)
])

network.train(mse, mse_derivative, X, Y, 10000, 0.01, True)

count = 0
size = 100
test_set_x1 = np.random.randint(0, 2, size=size)
test_set_x2 = np.random.randint(0, 2, size=size)
for x1, x2 in zip(test_set_x1, test_set_x2):

    print(x1, x2)
    y = network.predict([[x1], [x2]])
    print(y)

    if x1 == x1 and y <= 0.05:
        count += 1
    
    if x1 != x2 and y >= 0.95:
        count += 1
    
print(count)