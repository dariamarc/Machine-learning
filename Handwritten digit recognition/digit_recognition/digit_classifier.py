import sklearn.datasets
import numpy as np

from kmeans.kmeans import KMeans
from neural_net.neural_network import NeuralNetwork


class DigitClassifier:

    def __init__(self):
        print("Loading MNIST dataset...")
        data = sklearn.datasets.load_digits()
        self.inputs = data.data
        self.targets = data.target
        print("OK - Dataset loaded succesfully")
        self.neural_net = NeuralNetwork()
        self.kmeans = KMeans()

    def split_data(self, train_percentage=0.8):
        np.random.seed(5)
        indexes = [i for i in range(len(self.inputs))]
        train_sample = np.random.choice(indexes, int(train_percentage * len(self.inputs)), replace=False)
        test_sample = [i for i in indexes if i not in train_sample]

        train_inputs = [self.inputs[i] for i in train_sample]
        train_targets = [self.targets[i] for i in train_sample]

        test_inputs = [self.inputs[i] for i in test_sample]
        test_targets = [self.targets[i] for i in test_sample]

        return train_inputs, train_targets, test_inputs, test_targets

    def train_model(self, model):
        train_inputs, train_targets, test_inputs, test_targets = self.split_data()
        if model == "neural_net":
            print("Training neural network...")
            self.neural_net.train(train_inputs, train_targets)
        elif model == "kmeans":
            print("Training KMeans...")
            self.kmeans.train(data=train_inputs, no_clusters=10)
