"""Template for a 3 layer feed forward neural network for digit
classification, implemented from scratch.

Architecture: input -> hidden1 -> hidden2 -> output.

Implement the forward pass, back propagation, and weight update
yourself. You may use numpy for linear algebra. You may not use torch,
tensorflow, sklearn, jax, or keras for training, gradients, or
prediction.

Required public API (fixed for auto grading):
  * class `ScratchNeuralNetworkDigits` with methods `forward`,
    `backward`, `update_weights`, `train`, `predict`, `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)`.

Usage:
    python3 q1b_neural_net_scratch_digits.py <training_percent>
"""

import sys
import time
import numpy as np

from util_digits import load_digits, flatten_images


def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class ScratchNeuralNetworkDigits:
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64,
                 output_size=10, learning_rate=0.01, num_epochs=20,
                 batch_size=32, seed=None):
        rng = np.random.default_rng(seed)
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # He initialisation
        self.W1 = rng.standard_normal((input_size, hidden1_size)) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden1_size)
        self.W2 = rng.standard_normal((hidden1_size, hidden2_size)) * np.sqrt(2 / hidden1_size)
        self.b2 = np.zeros(hidden2_size)
        self.W3 = rng.standard_normal((hidden2_size, output_size)) * np.sqrt(2 / hidden2_size)
        self.b3 = np.zeros(output_size)
        self.cache = {}

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        y_hat = softmax(z3)
        self.cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "y_hat": y_hat}
        return y_hat

    def backward(self, X, y_onehot):
        N = X.shape[0]
        c = self.cache
        # Output layer gradient (softmax + cross-entropy)
        dz3 = (c["y_hat"] - y_onehot) / N
        dW3 = c["a2"].T @ dz3
        db3 = dz3.sum(axis=0)
        # Hidden layer 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_grad(c["z2"])
        dW2 = c["a1"].T @ dz2
        db2 = dz2.sum(axis=0)
        # Hidden layer 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(c["z1"])
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    def update_weights(self, grads):
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]
        self.W3 -= self.lr * grads["dW3"]
        self.b3 -= self.lr * grads["db3"]

    def train(self, training_images, training_labels):
        X = flatten_images(training_images)
        N = X.shape[0]
        # One-hot encode
        Y = np.zeros((N, 10))
        Y[np.arange(N), training_labels] = 1.0

        for _ in range(self.num_epochs):
            perm = np.random.permutation(N)
            X, Y = X[perm], Y[perm]
            for start in range(0, N, self.batch_size):
                Xb = X[start:start + self.batch_size]
                Yb = Y[start:start + self.batch_size]
                self.forward(Xb)
                grads = self.backward(Xb, Yb)
                self.update_weights(grads)

    def predict(self, image):
        x = image.ravel()[np.newaxis, :]
        return int(np.argmax(self.forward(x)))

    def evaluate(self, images, labels):
        X = flatten_images(images)
        probs = self.forward(X)
        preds = np.argmax(probs, axis=1)
        return float(np.mean(preds == labels))


def main(training_percent: int, num_iterations: int = 5) -> dict:
    training_images, training_labels = load_digits("training")
    test_images, test_labels = load_digits("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        net = ScratchNeuralNetworkDigits()
        start = time.time()
        net.train(training_images[idx], training_labels[idx])
        train_times[i] = time.time() - start
        accuracies[i] = net.evaluate(test_images, test_labels)

    errors = 1.0 - accuracies
    results = {
        "training_percent": training_percent,
        "mean_train_time": float(np.mean(train_times)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }
    print(f"\n=== Scratch NN | Digits | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)