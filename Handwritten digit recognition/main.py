from digit_recognition.digit_classifier import DigitClassifier


def main():
    digit_classifier = DigitClassifier()
    digit_classifier.train_model(model="kmeans")


main()