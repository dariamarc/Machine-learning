import numpy as np


# Base class for the layer
class Layer:
    def __init__(self):
        # each layer needs to have input data and to give an output
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


# Fully Convolutional Layer - these will be the layers used by the network
class FCLayer(Layer):
    # input_size - number of input neurons
    # output_size - number of output neurons
    def __init__(self, input_size, output_size):
        # initialize the weights and biases
        self.__weights = np.random.rand(input_size, output_size) - 0.5
        self.__bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.__weights) + self.__bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.__weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.__weights -= learning_rate * weights_error
        self.__bias -= learning_rate * output_error
        return input_error


# Activation Layer - this will be the layer that will apply the activation function to the input
class ActivationLayer(Layer):
    def __init__(self, activation, activation_dx):
        self.activation = activation
        self.activation_dx = activation_dx

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # we don't have any parameters to update
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_dx(self.input) * output_error


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_dx = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set the loss function
    def set_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_dx = loss_prime

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):

        sample_length = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(sample_length):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_dx(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= sample_length
            print('Epoch %d/%d   error=%f' % (i + 1, epochs, err))

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
