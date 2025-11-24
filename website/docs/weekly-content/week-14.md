---
title: Week 14 - Matrix Calculus for ML
sidebar_label: Week 14 - Matrix Calculus
---

# Week 14: Matrix Calculus for Machine Learning

## Time Allocation
**Total**: 13 hours | Theory: 5h | Coding: 4h | Project: 4h

## Learning Objectives
- Master derivatives of matrix expressions
- Understand backpropagation mathematics
- Learn Jacobian and Hessian matrices
- Apply to neural network training

## Core Concepts
- **Gradient**: ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ
- **Jacobian**: J_ij = ∂f_i/∂x_j
- **Hessian**: H_ij = ∂²f/∂x_i∂x_j
- **Chain Rule**: ∂f/∂x = (∂f/∂y)(∂y/∂x)

## NumPy Coding

```python
import numpy as np

# Numerical gradient
def numerical_gradient(f, x, eps=1e-5):
    """Compute gradient numerically"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# Test
def f(x):
    return x[0]**2 + 2*x[1]**2

x = np.array([3.0, 4.0])
num_grad = numerical_gradient(f, x)
analytical_grad = np.array([2*x[0], 4*x[1]])

print(f"Numerical gradient: {num_grad}")
print(f"Analytical gradient: {analytical_grad}")
print(f"Match: {np.allclose(num_grad, analytical_grad)}")

# Jacobian matrix
def jacobian(f, x, eps=1e-5):
    """Compute Jacobian matrix numerically"""
    n = len(x)
    f0 = f(x)
    m = len(f0) if hasattr(f0, '__len__') else 1

    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        J[:, j] = (f(x_plus) - f0) / eps

    return J

# Test with vector-valued function
def g(x):
    return np.array([x[0]**2 + x[1], x[0]*x[1]])

x = np.array([2.0, 3.0])
J = jacobian(g, x)
print(f"\nJacobian:\n{J}")
```

## Backpropagation from Scratch

```python
class SimpleNN:
    """2-layer neural network with backpropagation"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        """Forward pass"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, learning_rate=0.01):
        """Backward pass using chain rule"""
        m = X.shape[0]

        # Forward
        output = self.forward(X)

        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0)

        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0)

        # Update parameters
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# Test
X = np.random.randn(100, 2)
y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int).reshape(-1, 1)

model = SimpleNN(2, 10, 1)
for epoch in range(1000):
    model.backward(X, y, learning_rate=0.1)
    if epoch % 100 == 0:
        pred = model.forward(X)
        loss = np.mean((pred - y)**2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

## Resources
- [Matrix Calculus - Stanford CS231n](http://cs231n.stanford.edu/handouts/derivatives.pdf)
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
