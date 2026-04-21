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
    """Binary perceptron: predicts 1 for face, 0 for not a face.

    Implementation notes:
      * Keep a single weight matrix and a bias scalar.
      * For image x, compute `s = w . x + b` and predict 1 if s >= 0
        else 0.
      * On a mistake, update w and b in the direction that would have
        corrected the prediction.
    """

    def __init__(self, image_shape=(70, 60), max_iterations: int = 3):
        """Initialise weights and bias.

        `image_shape` is (rows, cols) for each input image.
        `max_iterations` is the number of full passes over the training
        set during `train`.
        """
        # TODO: initialize self.weights (shape: rows x cols) and self.bias.
        raise NotImplementedError

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        """Fit the perceptron on training data.

        `training_images` has shape (N, 70, 60). `training_labels` has
        shape (N,) with values in {0, 1}.
        """
        # TODO: implement the binary perceptron update rule.
        raise NotImplementedError

    def predict(self, image: np.ndarray) -> int:
        """Predict 0 or 1 for a single 70x60 image."""
        # TODO: return 1 if w . x + b >= 0 else 0.
        raise NotImplementedError

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy in [0, 1] over a batch."""
        # TODO: loop over images, call self.predict, compare with labels.
        raise NotImplementedError


def main(training_percent: int, num_iterations: int = 5) -> dict:
    """Run the standard train/test pipeline for the face perceptron.

    See `perceptron_digits.main` for protocol details. This variant
    uses the face dataset and the binary `PerceptronFacesClassifier`.
    """
    training_images, training_labels = load_faces("train")
    test_images, test_labels = load_faces("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        x_sample = training_images[idx]
        y_sample = training_labels[idx]

        clf = PerceptronFacesClassifier()
        start = time.time()
        clf.train(x_sample, y_sample)
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
