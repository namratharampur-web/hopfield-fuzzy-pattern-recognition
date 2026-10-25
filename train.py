import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(__file__), "src")
)

from hopfield import HopfieldNetwork
from dataset import get_patterns


def main():

    patterns = get_patterns()

    print("Characters stored in memory:")
    print(list(patterns.keys()))

    # 8 x 8 = 64 neurons
    num_neurons = 64

    # Convert matrices into vectors
    training_patterns = []

    for character, pattern in patterns.items():

        vector = pattern.flatten()

        training_patterns.append(vector)

    # Create Hopfield network
    network = HopfieldNetwork(num_neurons)

    # Train
    network.train(training_patterns)

    print("\nTraining completed.")

    print("\nWeight matrix shape:")
    print(network.weights.shape)

    # Save weights
    os.makedirs("results/outputs", exist_ok=True)

    import numpy as np

    np.save(
        "results/outputs/weights.npy",
        network.weights
    )

    print("\nWeights saved successfully.")


if __name__ == "__main__":
    main()