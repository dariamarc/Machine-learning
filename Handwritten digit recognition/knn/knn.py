import sklearn.neighbors


class Knn:

    def __init__(self):
        self.model = sklearn.neighbors.KNeighborsClassifier()

    def train(self, inputs, targets):
        self.model.fit(inputs, targets)

    def test(self, inputs):
        return self.model.predict(inputs)