---
title: Week 11 - QR Decomposition & Gram-Schmidt
sidebar_label: Week 11 - QR Decomposition
---

# Week 11: QR Decomposition & Gram-Schmidt

## Time Allocation
- **Total Hours**: 12 hours
- **Theory**: 4 hours | **NumPy Coding**: 4 hours | **Project**: 4 hours

## Learning Objectives
- Master QR decomposition
- Implement Gram-Schmidt orthogonalization
- Solve least squares problems
- Apply to regression and curve fitting

## Core Concepts
- **QR Decomposition**: A = QR where Q is orthogonal, R is upper triangular
- **Gram-Schmidt**: Convert basis to orthonormal basis
- **Applications**: Least squares, eigenvalue algorithms

## NumPy Coding

```python
import numpy as np

# QR decomposition
A = np.array([[1, 2], [3, 4], [5, 6]])
Q, R = np.linalg.qr(A)

print(f"Q (orthogonal):\n{Q}")
print(f"R (upper triangular):\n{R}")
print(f"QR = A? {np.allclose(Q @ R, A)}")
print(f"Q^T Q = I? {np.allclose(Q.T @ Q, np.eye(Q.shape[1]))}")

# Gram-Schmidt implementation
def gram_schmidt(V):
    """Orthonormalize columns of V"""
    n, m = V.shape
    U = np.zeros((n, m))

    for i in range(m):
        U[:, i] = V[:, i]
        for j in range(i):
            U[:, i] -= np.dot(U[:, j], V[:, i]) * U[:, j]
        U[:, i] /= np.linalg.norm(U[:, i])

    return U

# Least squares with QR
def solve_least_squares(A, b):
    """Solve Ax = b in least squares sense using QR"""
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b)

# Example: polynomial fitting
x = np.linspace(0, 10, 50)
y = 2*x**2 + 3*x + 5 + np.random.randn(50) * 10

# Build design matrix for quadratic
A = np.column_stack([np.ones_like(x), x, x**2])

# Solve
coeffs = solve_least_squares(A, y)
print(f"\nFitted coefficients: {coeffs}")

import matplotlib.pyplot as plt
plt.scatter(x, y, alpha=0.5, label='Data')
plt.plot(x, A @ coeffs, 'r-', label='Fitted curve')
plt.legend()
plt.title('Polynomial Fitting with QR')
plt.show()
```

## Project: Polynomial Regression Engine

```python
class PolynomialRegression:
    """Polynomial regression using QR decomposition"""

    def __init__(self, degree=2):
        self.degree = degree
        self.coefficients = None

    def fit(self, X, y):
        """Fit polynomial"""
        # Build Vandermonde matrix
        A = np.column_stack([X**i for i in range(self.degree + 1)])

        # Solve using QR
        Q, R = np.linalg.qr(A)
        self.coefficients = np.linalg.solve(R, Q.T @ y)

        return self

    def predict(self, X):
        """Predict"""
        A = np.column_stack([X**i for i in range(self.degree + 1)])
        return A @ self.coefficients

# Test
model = PolynomialRegression(degree=3)
model.fit(x, y)
predictions = model.predict(x)
```

## Resources
- [QR Decomposition - MIT](https://www.youtube.com/watch?v=FAnNBw7d0vg)
- [Gram-Schmidt Process](https://www.youtube.com/watch?v=zHbfZWZJTGc)
