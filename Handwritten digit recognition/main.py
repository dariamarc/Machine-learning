import time

from digit_recognition.digit_classifier import DigitClassifier


def main():
    digit_classifier = DigitClassifier()
    start_time = time.time()
    digit_classifier.train_model(model="knn")
    print("Training time: ", time.time() - start_time)
    digit_classifier.model_score(model="knn")

    # digit_classifier.train_model(model="neural_net")
    # print("Training time: ", time.time() - start_time)
    #
    # digit_classifier.model_score(model="neural_net")


main()