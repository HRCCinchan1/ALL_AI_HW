"""Template for a 3 layer feed forward neural network for binary face
classification, implemented with PyTorch.

This is Part 1(c) on the face task. You are expected to use `torch.nn`,
autograd, and `torch.optim`.

Required public API (fixed for auto grading):
  * class `PyTorchNeuralNetworkFaces` (a `torch.nn.Module` subclass)
    with a `forward` method.
  * class `PyTorchFacesClassifier` wrapper with `train`, `predict`,
    `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)`.

Usage:
    python3 q1c_neural_net_pytorch_faces.py <training_percent>
"""

import sys
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this file. Install with `pip install torch`."
    ) from exc

from util_faces import load_faces, flatten_images


class PyTorchNeuralNetworkFaces(nn.Module):
    """Three layer MLP: 4200 to hidden1 to hidden2 to 2 (face, not face)."""

    def __init__(self, input_size: int = 70 * 60,
                 hidden1_size: int = 128,
                 hidden2_size: int = 64,
                 output_size: int = 2):
        """Construct `nn.Linear` and activation modules for each layer."""
        super().__init__()
        # TODO: define self.fc1, self.fc2, self.fc3 and an activation.
        raise NotImplementedError

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass returning raw logits of shape (N, 2)."""
        # TODO: return self.fc3(act(self.fc2(act(self.fc1(x)))))
        raise NotImplementedError


class PyTorchFacesClassifier:
    """Thin wrapper that drives training and prediction for the module."""

    def __init__(
        self,
        hidden1_size: int = 128,
        hidden2_size: int = 64,
        learning_rate: float = 1e-3,
        num_epochs: int = 20,
        batch_size: int = 32,
        device: str | None = None,
    ):
        """Build the module, the loss, and the optimiser."""
        # TODO:
        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = PyTorchNeuralNetworkFaces(...).to(self.device)
        # self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        raise NotImplementedError

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        """Fit the PyTorch model on the provided training data."""
        # TODO: convert numpy to tensors, DataLoader, loop over epochs.
        raise NotImplementedError

    def predict(self, image: np.ndarray) -> int:
        """Predict 0 or 1 for a single 70x60 image."""
        # TODO: flatten, tensor, forward, argmax, return int.
        raise NotImplementedError

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy on a batch of images."""
        # TODO: vectorised eval in torch.no_grad() mode.
        raise NotImplementedError


def main(training_percent: int, num_iterations: int = 5) -> dict:
    """Run the standard train/test pipeline for the PyTorch NN on faces."""
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

        clf = PyTorchFacesClassifier()
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

    print(f"\n=== PyTorch NN | Faces | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)
