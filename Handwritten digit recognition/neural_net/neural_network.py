import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


class NeuralNetwork:

    def __init__(self, hidden_layer_size, alpha, solver, learning_rate):
        self.model = MLPClassifier(verbose=10, hidden_layer_sizes=hidden_layer_size, alpha=alpha, solver=solver, learning_rate_init=learning_rate, random_state=1)


    def train(self, inputs, targets, no_epochs=100):
        self.model.max_iter = no_epochs

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            self.model.fit(inputs, targets)

    def test(self, inputs):
        return self.model.predict(inputs)
