
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from hopfield_utils import (
    get_patterns,
    train_weights,
    add_noise,
    recall,
    pixel_accuracy,
)

st.set_page_config(page_title="Hopfield Pattern Recognition", page_icon="🧠")
st.title("Hopfield Network Pattern Recognition")
st.write("See how a Hopfield network reconstructs noisy character patterns.")

labels, patterns = get_patterns()
weights = train_weights(patterns)

character = st.selectbox("Choose a character", labels)
noise = st.slider("Noise level (%)", 0, 60, 25, 5)
seed = st.number_input("Noise seed", min_value=0, value=7)

original = patterns[labels.index(character)]
noisy = add_noise(original, noise / 100, seed=int(seed))
restored, energies = recall(noisy, weights, seed=int(seed))

cols = st.columns(3)
for col, data, name in zip(
    cols, [original, noisy, restored],
    ["Original", "Noisy input", "Reconstructed"]
):
    with col:
        fig, ax = plt.subplots()
        ax.imshow(data.reshape(8, 8), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(name)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

st.metric("Pixel accuracy", f"{pixel_accuracy(original, restored):.2f}%")
st.metric("Recall iterations", len(energies) - 1)
st.metric("Final energy", f"{energies[-1]:.2f}")

st.subheader("Noise vs accuracy")
noise_levels = [0, 10, 20, 30, 40, 50, 60]
scores = []

for percent in noise_levels:
    trials = []
    for i, pattern in enumerate(patterns):
        damaged = add_noise(pattern, percent / 100, seed=1000 + i)
        result, _ = recall(damaged, weights, seed=2000 + i)
        trials.append(pixel_accuracy(pattern, result))
    scores.append(np.mean(trials))

fig, ax = plt.subplots()
ax.plot(noise_levels, scores, marker="o")
ax.set_xlabel("Noise (%)")
ax.set_ylabel("Mean pixel accuracy (%)")
ax.set_ylim(0, 100)
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

st.caption("Educational example using five hand-coded patterns. Fuzzy logic is not included in this version.")
