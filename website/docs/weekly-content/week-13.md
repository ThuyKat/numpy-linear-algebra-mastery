---
title: Week 13 - Solving Linear Systems
sidebar_label: Week 13 - Solving Linear Systems
---

# Week 13: Solving Linear Systems

## Time Allocation
**Total**: 12 hours | Theory: 4h | Coding: 4h | Project: 4h

## Learning Objectives
- Master methods for solving Ax = b
- Understand iterative vs direct methods
- Learn condition numbers and stability
- Apply to engineering problems

## Core Concepts
- **Direct Methods**: LU, Cholesky, QR
- **Iterative Methods**: Jacobi, Gauss-Seidel, Conjugate Gradient
- **Condition Number**: κ(A) = ||A|| ||A⁻¹||
- **Ill-conditioned systems**: Small changes in b cause large changes in x

## NumPy Coding

```python
import numpy as np

# Direct solution
A = np.array([[3, 2], [1, 4]])
b = np.array([5, 6])

x = np.linalg.solve(A, b)
print(f"Solution: {x}")
print(f"Verification: {np.allclose(A @ x, b)}")

# Condition number
cond = np.linalg.cond(A)
print(f"Condition number: {cond:.2f}")

# Iterative: Jacobi method
def jacobi(A, b, x0, max_iter=100, tol=1e-6):
    """Jacobi iterative method"""
    D = np.diag(np.diag(A))
    R = A - D
    x = x0.copy()

    for i in range(max_iter):
        x_new = np.linalg.inv(D) @ (b - R @ x)
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged in {i+1} iterations")
            return x_new
        x = x_new

    return x

x_jacobi = jacobi(A, b, x0=np.zeros(2))
print(f"Jacobi solution: {x_jacobi}")
```

## Project: Sparse Linear Systems

```python
from scipy.sparse import csr_matrix, linalg as sp_linalg

# Large sparse system
n = 1000
A_sparse = sp_linalg.LaplacianNd((n,))
b_sparse = np.random.rand(n)

# Solve with conjugate gradient
x, info = sp_linalg.cg(A_sparse, b_sparse)
print(f"CG converged: {info == 0}")
```

## Resources
- [Iterative Methods - MIT](https://www.youtube.com/watch?v=mbK6i5eNmHk)
- [Condition Numbers](https://nhigham.com/2020/03/19/what-is-a-condition-number/)
