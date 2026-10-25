"""Shared utilities for a small Hopfield pattern-recognition project."""
import numpy as np

PATTERNS = {
    "A": ["00111100","01100110","11000011","11000011","11111111","11000011","11000011","11000011"],
    "B": ["11111100","11000110","11000110","11111100","11000110","11000110","11000110","11111100"],
    "C": ["00111110","01100011","11000000","11000000","11000000","11000000","01100011","00111110"],
    "D": ["11111000","11001100","11000110","11000110","11000110","11000110","11001100","11111000"],
    "E": ["11111111","11000000","11000000","11111100","11000000","11000000","11000000","11111111"],
}

def get_patterns():
    labels = list(PATTERNS)
    vectors = []
    for label in labels:
        bits = np.array([[int(c) for c in row] for row in PATTERNS[label]])
        vectors.append(np.where(bits.reshape(-1) == 1, 1, -1))
    return labels, np.array(vectors)

def train_weights(patterns):
    n = patterns.shape[1]
    weights = np.zeros((n, n), dtype=float)
    for pattern in patterns:
        weights += np.outer(pattern, pattern)
    weights /= n
    np.fill_diagonal(weights, 0)
    return weights

def energy(state, weights):
    return float(-0.5 * state @ weights @ state)

def recall(noisy_state, weights, max_iterations=20, seed=42):
    state = noisy_state.copy().astype(int)
    rng = np.random.default_rng(seed)
    energies = [energy(state, weights)]
    for _ in range(max_iterations):
        previous = state.copy()
        for idx in rng.permutation(len(state)):
            net = weights[idx] @ state
            if net > 0:
                state[idx] = 1
            elif net < 0:
                state[idx] = -1
        energies.append(energy(state, weights))
        if np.array_equal(state, previous):
            break
    return state, energies

def add_noise(pattern, noise_level, seed=None):
    if not 0 <= noise_level <= 1:
        raise ValueError("noise_level must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    noisy = pattern.copy()
    count = int(round(noise_level * len(pattern)))
    if count:
        noisy[rng.choice(len(pattern), size=count, replace=False)] *= -1
    return noisy

def pixel_accuracy(expected, actual):
    return float(np.mean(expected == actual) * 100)
