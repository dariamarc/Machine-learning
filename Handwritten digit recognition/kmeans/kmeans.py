import sklearn.cluster


class KMeans:

    def __init__(self):
        self.classifier = sklearn.cluster.KMeans()

    def train(self, data, no_clusters, no_epochs=100):
        self.classifier.n_clusters = no_clusters
        self.classifier.max_iter = no_epochs
        self.classifier.random_state = 0

        self.classifier.fit(data)

    def test(self):
        None