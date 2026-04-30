"""Template for the multi class Perceptron digit classifier.

Implement the perceptron training and prediction logic yourself. Do
not call sklearn, torch, or any other library's perceptron.

Required public API (fixed for auto grading):
  * class `PerceptronDigitsClassifier` with methods `train`, `predict`,
    `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)` which runs
    the full train/test pipeline and prints results in the standard
    format below.

Usage:
    python3 q1a_perceptron_digits.py <training_percent>
    e.g.  python3 q1a_perceptron_digits.py 50
"""

import sys
import time
import numpy as np

from util_digits import load_digits


class PerceptronDigitsClassifier:
    def __init__(self, num_classes: int = 10, image_shape=(28, 28), max_iterations: int = 3):
        self.num_classes = num_classes
        self.max_iterations = max_iterations
        rows, cols = image_shape
        self.weights = np.zeros((num_classes, rows * cols))
        self.biases = np.zeros(num_classes)

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        X = training_images.reshape(len(training_images), -1)
        for _ in range(self.max_iterations):
            for x, label in zip(X, training_labels):
                scores = self.weights @ x + self.biases
                pred = int(np.argmax(scores))
                if pred != label:
                    self.weights[label] += x
                    self.biases[label] += 1
                    self.weights[pred] -= x
                    self.biases[pred] -= 1

    def predict(self, image: np.ndarray) -> int:
        x = image.ravel()
        scores = self.weights @ x + self.biases
        return int(np.argmax(scores))

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        preds = [self.predict(img) for img in images]
        return float(np.mean(np.array(preds) == labels))


def main(training_percent: int, num_iterations: int = 5) -> dict:
    training_images, training_labels = load_digits("training")
    test_images, test_labels = load_digits("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        clf = PerceptronDigitsClassifier()
        start = time.time()
        clf.train(training_images[idx], training_labels[idx])
        train_times[i] = time.time() - start
        accuracies[i] = clf.evaluate(test_images, test_labels)

    errors = 1.0 - accuracies
    results = {
        "training_percent": training_percent,
        "mean_train_time": float(np.mean(train_times)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }
    print(f"\n=== Perceptron | Digits | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)
