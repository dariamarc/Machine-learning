from digit_recognition.digit_classifier import DigitClassifier


def main():
    digit_classifier = DigitClassifier()
    digit_classifier.train_model(model="knn")
    digit_classifier.model_score(model="knn")


main()