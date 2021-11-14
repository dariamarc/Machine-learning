import pandas
import sklearn.datasets
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from kmeans.kmeans import KMeans
from knn.knn import Knn
from neural_net.neural_network import NeuralNetwork


class DigitClassifier:

    def __init__(self):
        print("Loading MNIST dataset...")
        # data = sklearn.datasets.load_digits()
        # self.inputs = data.data
        # self.targets = data.target
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
        self.inputs = pandas.DataFrame.to_numpy(X / 255.0)
        self.targets = pandas.Series.to_numpy(y)
        print("OK - Dataset loaded succesfully")
        self.neural_net = NeuralNetwork((50,), 1e-4, "sgd", 0.1)
        self.kmeans = KMeans()
        self.knn = Knn()

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
        self.train_inputs, self.train_targets, self.test_inputs, self.test_targets = self.split_data()
        if model == "neural_net":
            print("Training neural network...")
            self.neural_net.train(self.train_inputs, self.train_targets)
        elif model == "kmeans":
            print("Training KMeans...")
            self.kmeans.train(data=self.train_inputs, no_clusters=10)
        elif model == "knn":
            print("Training KNN...")
            self.knn.train(self.train_inputs, self.train_targets)

    def test_model(self, model):
        if model == "neural_net":
            print("Predicting data using neural network...")
            return self.neural_net.test(self.test_inputs)
        elif model == "kmeans":
            print("Predicting data using KMeans...")
            return self.kmeans.test(self.test_inputs)
        elif model == "knn":
            print("Predicting data using KNN...")
            return self.knn.test(self.test_inputs)

    def model_score(self, model):
        computed = self.test_model(model)
        print("Accuracy: ", accuracy_score(self.test_targets, computed))
