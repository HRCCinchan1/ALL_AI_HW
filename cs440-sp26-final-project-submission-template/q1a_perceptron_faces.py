"""Template for the binary Perceptron face classifier.

Implement the perceptron training and prediction logic yourself. Do
not call sklearn, torch, or any other library's perceptron.

Required public API (fixed for auto grading):
  * class `PerceptronFacesClassifier` with methods `train`, `predict`,
    `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)` which runs
    the full train/test pipeline and prints results in the standard
    format below.

Usage:
    python3 q1a_perceptron_faces.py <training_percent>
    e.g.  python3 q1a_perceptron_faces.py 50
"""

import sys
import time
import numpy as np

from util_faces import load_faces


class PerceptronFacesClassifier:
    def __init__(self, image_shape=(70, 60), max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.weights = np.zeros(image_shape)
        self.bias = 0.0

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        for _ in range(self.max_iterations):
            for img, label in zip(training_images, training_labels):
                pred = self.predict(img)
                if pred != label:
                    error = label - pred  # +1 or -1
                    self.weights += error * img
                    self.bias += error

    def predict(self, image: np.ndarray) -> int:
        score = np.dot(self.weights.ravel(), image.ravel()) + self.bias
        return 1 if score >= 0 else 0

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        correct = sum(self.predict(img) == lbl for img, lbl in zip(images, labels))
        return correct / len(labels)


def main(training_percent: int, num_iterations: int = 5) -> dict:
    training_images, training_labels = load_faces("train")
    test_images, test_labels = load_faces("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        clf = PerceptronFacesClassifier()
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
    print(f"\n=== Perceptron | Faces | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)