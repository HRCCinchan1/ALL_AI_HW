import json
import matplotlib.pyplot as plt

with open("results.json") as f:
    data = json.load(f)

fractions = [d["training_percent"] for d in data["perceptron_digits"]]

def plot(metric, title):
    for model in data:
        vals = [d[metric] for d in data[model]]
        plt.plot(fractions, vals, label=model)

    plt.title(title)
    plt.xlabel("Training %")
    plt.legend()
    plt.show()

plot("mean_accuracy", "Accuracy vs Training %")
plot("mean_error", "Error vs Training %")
plot("mean_train_time", "Training Time vs Training %")