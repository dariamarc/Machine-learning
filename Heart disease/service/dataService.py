
import numpy as np

from repository.repository import Repository


class DataService:

    def __init__(self):
        self.__repo = Repository()

    def import_data(self, filename):
        header, data = self.__repo.load_data("heart.csv")
        print("Working with following features: ")
        print(header)

        inputs, outputs = self.__extract_data(data)

        inputs = self.__to_numerical_list(inputs)
        outputs = list(map(int, outputs))

        return inputs, outputs

    def __extract_data(self, data):
        inputs = []
        outputs = []

        for row in data:
            inputs.append(row[:-1])
            outputs.append(row[-1])

        return inputs, outputs

    def __to_numerical_list(self, string_list):

        numerical_list = []

        for x in string_list:
            numerical_list.append(list(map(float, x)))

        return numerical_list

    def preprocess_data(self, inputs, outputs):

        inputs, outputs = self.__shuffle_data(inputs, outputs)

        train_inputs, train_outputs, test_inputs, test_outputs = self.__split_data(inputs, outputs)

        train_inputs, test_inputs = self.__normalize(train_inputs, test_inputs)
        # test_inputs = self.__normalize(test_inputs)

        return train_inputs, train_outputs, test_inputs, test_outputs


    def __shuffle_data(self, inputs, outputs):

        data_len = len(inputs)
        permutation = np.random.permutation(data_len)

        new_inputs = [inputs[i] for i in permutation]
        new_outputs = [outputs[i] for i in permutation]

        return new_inputs, new_outputs

    def __split_data(self, inputs, outputs):

        np.random.seed(5)
        indexes = [i for i in range(len(inputs))]
        train_sample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
        test_sample = [i for i in indexes if i not in train_sample]

        train_inputs = [inputs[i] for i in train_sample]
        train_outputs = [outputs[i] for i in train_sample]
        test_inputs = [inputs[i] for i in test_sample]
        test_outputs = [outputs[i] for i in test_sample]

        return train_inputs, train_outputs, test_inputs, test_outputs

    def __normalize(self, train_inputs, test_inputs):
        # normalize data that contains numerical values
        train_inputs, test_inputs = self.__scale(train_inputs, test_inputs)

        train_inputs, test_inputs = self.__center(train_inputs, test_inputs)

        return train_inputs, test_inputs


    def __scale(self, train_inputs, test_inputs):

        non_numerical_idxes = [1, 2, 5, 6, 8, 10, 11, 12]
        numerical_idxes = [i for i in range(13) if i not in non_numerical_idxes]

        for idx in numerical_idxes:
            # scale each feature
            input_features = []
            for input in train_inputs:
                input_features.append(input[idx])

            min_value = min(input_features)
            max_value = max(input_features)

            for i in range(len(train_inputs)):
                train_inputs[i][idx] = (train_inputs[i][idx] - min_value) / (max_value - min_value)

            for i in range(len(test_inputs)):
                test_inputs[i][idx] = (test_inputs[i][idx] - min_value) / (max_value - min_value)

        return train_inputs, test_inputs

    def __center(self, train_inputs, test_inputs):

        non_numerical_idxes = [1, 2, 5, 6, 8, 10, 11, 12]
        numerical_idxes = [i for i in range(13) if i not in non_numerical_idxes]

        for idx in numerical_idxes:
            # center each feature
            input_features = []
            for input in train_inputs:
                input_features.append(input[idx])

            mean_value = sum(input_features) / len(input_features)

            for i in range(len(train_inputs)):
                train_inputs[i][idx] = train_inputs[i][idx] - mean_value

            for i in range(len(test_inputs)):
                test_inputs[i][idx] = test_inputs[i][idx] - mean_value

        return train_inputs, test_inputs
