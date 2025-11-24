---
title: Week 16 - Neural Networks Mathematics
sidebar_label: Week 16 - Neural Networks
---

# Week 16: Neural Networks Mathematics

## Time Allocation
**Total**: 14 hours | Theory: 5h | Coding: 5h | Project: 4h

## Learning Objectives
- Understand forward and backward propagation
- Master weight initialization techniques
- Learn batch normalization mathematics
- Implement multilayer networks from scratch

## Core Concepts
- **Forward Pass**: z = Wx + b, a = σ(z)
- **Backward Pass**: Chain rule through layers
- **Xavier Init**: Var(W) = 2/(n_in + n_out)
- **Batch Norm**: Normalize activations per batch

## NumPy Coding

```python
import numpy as np

class Layer:
    """Base neural network layer"""

    def forward(self, X):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

class Dense(Layer):
    """Fully connected layer"""

    def __init__(self, input_dim, output_dim):
        # Xavier initialization
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2/(input_dim + output_dim))
        self.b = np.zeros(output_dim)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dout, learning_rate=0.01):
        dX = dout @ self.W.T
        dW = self.X.T @ dout
        db = np.sum(dout, axis=0)

        # Update parameters
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dX

class ReLU(Layer):
    """ReLU activation"""

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dout, learning_rate=None):
        return dout * (self.X > 0)

class Softmax(Layer):
    """Softmax activation"""

    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.output

    def backward(self, y_true, learning_rate=None):
        return self.output - y_true

# Build network
class NeuralNetwork:
    """Multilayer neural network"""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, learning_rate=0.01):
        dout = self.layers[-1].backward(y_true)
        for layer in reversed(self.layers[:-1]):
            dout = layer.backward(dout, learning_rate)

    def train(self, X, y, epochs=100, batch_size=32):
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Forward and backward
                output = self.forward(X_batch)
                self.backward(y_batch)

            if epoch % 10 == 0:
                output = self.forward(X)
                loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
                acc = np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1))
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Acc = {acc:.4f}")

# Test on XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoded

model = NeuralNetwork([
    Dense(2, 4),
    ReLU(),
    Dense(4, 2),
    Softmax()
])

model.train(X, y, epochs=1000, batch_size=4)
```

## Project: MNIST from Scratch

```python
# Simplified MNIST implementation
def load_mnist_subset():
    """Load small subset of MNIST"""
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data / 16.0  # Normalize
    y = np.eye(10)[digits.target]  # One-hot
    return X, y

X, y = load_mnist_subset()

# Split train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build deeper network
model = NeuralNetwork([
    Dense(64, 128),
    ReLU(),
    Dense(128, 64),
    ReLU(),
    Dense(64, 10),
    Softmax()
])

model.train(X_train, y_train, epochs=100, batch_size=32)

# Evaluate
output = model.forward(X_test)
accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(y_test, axis=1))
print(f"\nTest Accuracy: {accuracy:.4f}")
```

## Resources
- [CS231n - Neural Networks](http://cs231n.github.io/neural-networks-1/)
- [Deep Learning Book - Chapter 6](https://www.deeplearningbook.org/contents/mlp.html)
