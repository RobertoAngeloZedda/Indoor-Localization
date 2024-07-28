class Network():
    
    def __init__(self, layers):
        self.layers = layers
    
    def predict(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def train(self, loss, loss_derivative, X, Y, epochs, learning_rate, print_debug):
        for epoche in range(epochs):
            error = 0

            i = 0
            for x, y_real in zip(X, Y):

                y = self.predict(x)

                error += loss(y_real, y)

                gradient = loss_derivative(y_real, y)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)
                
                if print_debug:
                    print(f'{epoche+1}/{epochs}\t{(i+1) * 100 / len(X)}%\tError: {error / (i+1)}')
                i +=1

            error /= len(X)

            if print_debug:
                print(f'{(epoche+1) * 100 / epochs}%\t\tError = {error}')
    
    def train_stocastic(self, loss, loss_derivative, X, Y, epochs, batches, learning_rate, print_debug):
        for epoche in range(epochs):
            error = 0

            i = 0
            batch_counter = 0
            gradient_mean = 0
            for x, y_real in zip(X, Y):

                y = self.predict(x)

                error += loss(y_real, y)
                gradient_mean += loss_derivative(y_real, y)
                
                batch_counter += 1

                if batch_counter >= batches or x == X[-1]:
                    
                    gradient_mean /= batches

                    for layer in reversed(self.layers):
                        gradient_mean = layer.backward(gradient_mean, learning_rate)

                    batch_counter = 0
                    gradient_mean = 0

                    if print_debug:
                        print(f'{epoche+1}/{epochs}\t{(i+1) * 100 / len(X)}%\tError: {error / (i+1)}')
                
                i +=1

            error /= len(X)

            if print_debug:
                print(f'{(epoche+1) * 100 / epochs}%\t\tError = {error}')