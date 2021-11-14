import sklearn.cluster


class KMeans:

    def __init__(self):
        self.classifier = sklearn.cluster.KMeans(init="k-means++", verbose=1)

    def train(self, data, no_clusters, no_epochs=1000):
        self.classifier.n_clusters = no_clusters
        self.classifier.max_iter = no_epochs
        self.classifier.random_state = 0

        self.classifier.fit(data)

    def test(self, data):
        computed_outputs = self.classifier.predict(data)
        return computed_outputs