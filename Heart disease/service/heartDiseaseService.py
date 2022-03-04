from math import sqrt

import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score, auc
from service.neural_net import Network, FCLayer, ActivationLayer
from service.dataService import DataService

def relu(x):
    return np.maximum(x, 0)

def relu_dx(x):
    return np.greater(x, 0).astype(float)

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_dx(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

class HeartDiseaseService:

    def __init__(self):
        self.__data_service = DataService()

    def run_model(self):
        inputs, outputs = self.__data_service.import_data("heart.csv")

        train_inputs, train_outputs, test_inputs, test_outputs = self.__data_service.preprocess_data(inputs, outputs)

        x_train = np.asarray(train_inputs)
        x_train = np.resize(x_train, (242, 1, 13))
        y_train = np.asarray(train_outputs)
        y_train = np_utils.to_categorical(y_train)
        y_train = np.resize(y_train, (242, 1, 2))
        print(x_train.shape)
        print(y_train.shape)

        # build the network
        neuraln = Network()
        neuraln.add(FCLayer(13, 4))
        neuraln.add(ActivationLayer(relu, relu_dx))
        neuraln.add(FCLayer(4, 2))
        neuraln.add(ActivationLayer(relu, relu_dx))

        neuraln.set_loss(mse, mse_dx)
        # train the network
        neuraln.fit(x_train, y_train, epochs=100, learning_rate=0.1)

        # test the network
        x_test = np.asarray(test_inputs)
        x_test = np.resize(x_test, (len(x_test), 1, 13))

        results = neuraln.predict(x_test)

        predicted = []
        for x in results:
            predicted.append(0 if x[0][0] > x[0][1] else 1)


        results = np.resize(results, (len(results), 2))
        self.evaluate_network(test_outputs, predicted, results)


    def evaluate_network(self, target, predicted, probs):
        accuracy = accuracy_score(target, predicted)
        precision = precision_score(target, predicted)
        recall = recall_score(target, predicted)
        F1_score = f1_score(target, predicted)

        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 score: " + str(F1_score))

        # confidence interval for classification accuracy
        interval = 1.96 * sqrt( (accuracy * (1 - accuracy)) / len(target))
        print("Confidence interval accuracy radius: " + str(interval))

        # ROC AUC
        # keep probabilities for the positive outcome only
        lr_probs = probs[:, 1]

        lr_precision, lr_recall, _ = precision_recall_curve(target, lr_probs)
        lr_f1, lr_auc = F1_score, auc(lr_recall, lr_precision)
        # summarize scores
        print('Model classification: auc=%.3f' % lr_auc)
        # plot the precision-recall curves
        no_skill = target[target == 1] / len(target)
        pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No skill')
        pyplot.plot(lr_recall, lr_precision, marker='.', label='Classification')
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

