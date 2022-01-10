import sklearn.neighbors


class Knn:

    def __init__(self):
        self.model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)

    def train(self, inputs, targets):
        self.model.fit(inputs, targets)

    def train_K(self, n_neighbours, inputs, targets):
        self.model.n_neighbors = n_neighbours
        self.model.fit(inputs, targets)

    def test(self, inputs):
        print(self.model.n_neighbors)
        return self.model.predict(inputs)

