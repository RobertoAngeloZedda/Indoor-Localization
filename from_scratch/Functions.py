import numpy as np

def ReLU(x):
    if type(x) == np.ndarray:
        return np.reshape([ReLU(item) for item in x], x.shape)
    else:
        return max(x, 0)

def ReLU_derivative(x):
    if type(x) == np.ndarray:
        return np.reshape([ReLU(item) for item in x], x.shape)
    else:
        return 1 if x > 0 else 0

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# LOSS

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)