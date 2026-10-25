
import numpy as np

from src.hopfield import HopfieldNetwork
from src.dataset import get_patterns
from src.preprocessing import add_noise, calculate_accuracy


def display_pattern(pattern, title):
    print(f"\n{title}")

    grid = np.asarray(pattern).reshape(8, 8)

    for row in grid:
        print("".join("#" if pixel == 1 else "." for pixel in row))


def main():
    patterns = get_patterns()

    # Train using all five example characters.
    training_data = [
        pattern.flatten()
        for pattern in patterns.values()
    ]

    network = HopfieldNetwork(num_neurons=64)
    network.train(training_data)

    original = patterns["A"].flatten()
    noisy = add_noise(original, noise_level=0.20)

    reconstructed, iterations = network.predict(noisy)

    display_pattern(original, "Original character A")
    display_pattern(noisy, "Noisy character A")
    display_pattern(reconstructed, "Reconstructed character A")

    accuracy = calculate_accuracy(original, reconstructed)

    print(f"\nPixel accuracy: {accuracy:.2%}")
    print(f"Iterations: {iterations}")
    print(f"Final energy: {network.energy(reconstructed):.2f}")


if __name__ == "__main__":
    main()