"""A 2-layer neural network written from scratch with NumPy (no PyTorch / TensorFlow).

Architecture:  64 inputs (8x8 pixels) -> 64 hidden (ReLU) -> 10 outputs (softmax)
"""
import numpy as np


class NeuralNetwork:
    def __init__(self, n_in=64, n_hidden=64, n_out=10, seed=0):
        rng = np.random.default_rng(seed)
        # He initialisation keeps signal size stable through ReLU layers
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        # Adam optimiser state
        self.params = ["W1", "b1", "W2", "b2"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    # ---------- forward pass ----------
    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)                       # ReLU
        z2 = self.a1 @ self.W2 + self.b2
        z2 -= z2.max(axis=1, keepdims=True)                    # numerical stability
        exp = np.exp(z2)
        self.probs = exp / exp.sum(axis=1, keepdims=True)      # softmax
        return self.probs

    @staticmethod
    def loss(probs, y):
        """Cross-entropy: punishes confident wrong answers heavily."""
        return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))

    # ---------- backward pass (backpropagation) ----------
    def backward(self, y, l2=1e-4):
        n = len(y)
        d2 = self.probs.copy()
        d2[np.arange(n), y] -= 1                               # dLoss/dz2 for softmax + cross-entropy
        d2 /= n
        grads = {"W2": self.a1.T @ d2 + l2 * self.W2, "b2": d2.sum(0)}
        d1 = (d2 @ self.W2.T) * (self.z1 > 0)                  # chain rule through ReLU
        grads["W1"] = self.X.T @ d1 + l2 * self.W1
        grads["b1"] = d1.sum(0)
        return grads

    def step(self, grads, lr=0.003, b1=0.9, b2=0.999, eps=1e-8):
        """Adam: gradient descent with momentum + per-weight adaptive step size."""
        self.t += 1
        for p in self.params:
            self.m[p] = b1 * self.m[p] + (1 - b1) * grads[p]
            self.v[p] = b2 * self.v[p] + (1 - b2) * grads[p] ** 2
            m_hat = self.m[p] / (1 - b1 ** self.t)
            v_hat = self.v[p] / (1 - b2 ** self.t)
            setattr(self, p, getattr(self, p) - lr * m_hat / (np.sqrt(v_hat) + eps))

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def to_dict(self):
        return {p: np.round(getattr(self, p), 5).tolist() for p in self.params}
