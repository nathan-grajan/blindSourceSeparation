# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Create sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# Create two distinct signals (sine wave and square wave)
s1 = np.sin(2 * time)  # Sine wave
s2 = np.sign(np.sin(3 * time))  # Square wave

# Mix data
S = np.c_[s1, s2]
A = np.array([[1, 1], [0.5, 2]])  # Mixing matrix
X = S.dot(A.T)  # Generate observations

# Apply FastICA
ica = FastICA(n_components=2)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

plt.figure(figsize=(9, 8))

plt.subplot(4, 1, 1)
plt.title('True Source 1: Sine Wave')
plt.plot(time, s1, color='blue')

plt.subplot(4, 1, 2)
plt.title('True Source 2: Square Wave')
plt.plot(time, s2, color='orange')

plt.subplot(4, 1, 3)
plt.title('Mixed Signal')
plt.plot(time, X[:, 0], color='green')

plt.subplot(4, 1, 4)
plt.title('Separated Signal using ICA')
plt.plot(time, S_[:, 0], color='red')

plt.tight_layout()
plt.show()
# %%
