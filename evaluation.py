
import os
import numpy as np
import matplotlib.pyplot as plt

from src.hopfield import HopfieldNetwork
from src.dataset import get_patterns
from src.preprocessing import add_noise, calculate_accuracy


def main():
    np.random.seed(42)

    patterns = get_patterns()
    training_data = [
        pattern.flatten()
        for pattern in patterns.values()
    ]

    network = HopfieldNetwork(num_neurons=64)
    network.train(training_data)

    noise_levels = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    runs_per_level = 50
    average_accuracies = []

    for noise in noise_levels:
        accuracies = []

        for _ in range(runs_per_level):
            # Test each of the five characters.
            for original_grid in patterns.values():
                original = original_grid.flatten()
                noisy = add_noise(original, noise)
                reconstructed, _ = network.predict(noisy)

                accuracy = calculate_accuracy(
                    original, reconstructed
                )
                accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        average_accuracies.append(mean_accuracy)

        print(
            f"Noise: {noise:>5.0%} | "
            f"Average pixel accuracy: {mean_accuracy:.2%}"
        )

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(
        [level * 100 for level in noise_levels],
        [accuracy * 100 for accuracy in average_accuracies],
        marker="o"
    )

    plt.title("Hopfield Network: Noise vs Reconstruction Accuracy")
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Average Pixel Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()

    output_path = "results/plots/noise_vs_accuracy.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"\nGraph saved to: {output_path}")


if __name__ == "__main__":
    main()