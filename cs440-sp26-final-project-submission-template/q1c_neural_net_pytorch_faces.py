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
    raise ImportError("PyTorch is required. Install with `pip install torch`.") from exc

from util_faces import load_faces, flatten_images


class PyTorchNeuralNetworkFaces(nn.Module):
    def __init__(self, input_size=70*60, hidden1_size=128, hidden2_size=64, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


class PyTorchFacesClassifier:
    def __init__(self, hidden1_size=128, hidden2_size=64, learning_rate=1e-3,
                 num_epochs=20, batch_size=32, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchNeuralNetworkFaces(hidden1_size=hidden1_size,
                                               hidden2_size=hidden2_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def train(self, training_images, training_labels):
        X = torch.tensor(flatten_images(training_images), dtype=torch.float32)
        y = torch.tensor(training_labels, dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.num_epochs):
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(Xb), yb)
                loss.backward()
                self.optimizer.step()

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(image.ravel(), dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(self.model(x).argmax(dim=1).item())

    def evaluate(self, images, labels):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(flatten_images(images), dtype=torch.float32).to(self.device)
            preds = self.model(X).argmax(dim=1).cpu().numpy()
        return float(np.mean(preds == labels))


def main(training_percent: int, num_iterations: int = 5) -> dict:
    training_images, training_labels = load_faces("train")
    test_images, test_labels = load_faces("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        clf = PyTorchFacesClassifier()
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
    print(f"\n=== PyTorch NN | Faces | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)
