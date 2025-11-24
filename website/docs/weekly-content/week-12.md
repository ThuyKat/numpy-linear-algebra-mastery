---
title: Week 12 - Optimization & Gradient Descent
sidebar_label: Week 12 - Optimization & Gradient Descent
---

# Week 12: Optimization & Gradient Descent

## Time Allocation
- **Total Hours**: 13 hours
- **Theory**: 5 hours | **NumPy Coding**: 4 hours | **Project**: 4 hours

## Learning Objectives
- Understand gradient descent and variants
- Learn convex optimization basics
- Implement optimizers from scratch
- Apply to machine learning problems

## Core Concepts
- **Gradient**: Direction of steepest ascent ∇f(x)
- **Gradient Descent**: x_(t+1) = x_t - α∇f(x_t)
- **Variants**: SGD, Momentum, Adam
- **Convexity**: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)

## NumPy Coding

```python
import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    """Gradient descent optimizer"""

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def optimize(self, gradient_fn, x0):
        """Optimize using gradient descent"""
        x = x0.copy()

        for i in range(self.max_iter):
            grad = gradient_fn(x)
            x_new = x - self.lr * grad

            self.history.append(x.copy())

            if np.linalg.norm(x_new - x) < self.tol:
                print(f"Converged in {i+1} iterations")
                break

            x = x_new

        return x

# Example: minimize f(x) = x^2 + y^2
def gradient(x):
    return 2 * x

optimizer = GradientDescent(learning_rate=0.1)
x_min = optimizer.optimize(gradient, x0=np.array([5.0, 5.0]))
print(f"Minimum found at: {x_min}")

# Visualize optimization path
history = np.array(optimizer.history)

# Create contour plot
x_range = np.linspace(-6, 6, 100)
y_range = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 + Y**2

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=20, alpha=0.6)
plt.plot(history[:, 0], history[:, 1], 'r.-', linewidth=2, markersize=8)
plt.plot(history[0, 0], history[0, 1], 'go', markersize=12, label='Start')
plt.plot(history[-1, 0], history[-1, 1], 'r*', markersize=15, label='End')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Optimization Path')
plt.legend()
plt.grid(True, alpha=0.3)
plt.colorbar(label='f(x, y)')
plt.show()
```

### Momentum and Adam

```python
class MomentumOptimizer:
    """Gradient descent with momentum"""

    def __init__(self, learning_rate=0.01, momentum=0.9, max_iter=1000):
        self.lr = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter

    def optimize(self, gradient_fn, x0):
        x = x0.copy()
        velocity = np.zeros_like(x)

        for i in range(self.max_iter):
            grad = gradient_fn(x)
            velocity = self.momentum * velocity - self.lr * grad
            x = x + velocity

        return x

class AdamOptimizer:
    """Adam optimizer"""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, max_iter=1000):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iter = max_iter

    def optimize(self, gradient_fn, x0):
        x = x0.copy()
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment

        for t in range(1, self.max_iter + 1):
            grad = gradient_fn(x)

            # Update moments
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # Update parameters
            x = x - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return x
```

## Project: Linear Regression with Gradient Descent

```python
class LinearRegressionGD:
    """Linear regression trained with gradient descent"""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Compute loss
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            # Compute gradients
            dw = -(2 / n_samples) * X.T @ (y - y_pred)
            db = -(2 / n_samples) * np.sum(y - y_pred)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.squeeze() + np.random.randn(100)

# Train model
model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

# Plot results
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, model.predict(X), 'r-', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')

plt.subplot(1, 2, 2)
plt.plot(model.loss_history)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nLearned parameters: w={model.weights[0]:.2f}, b={model.bias:.2f}")
```

## Resources
- [3Blue1Brown - Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [Adam Paper](https://arxiv.org/abs/1412.6980)
