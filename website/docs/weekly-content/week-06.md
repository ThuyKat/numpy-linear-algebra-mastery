---
title: Week 6 - LU Decomposition
sidebar_label: Week 6 - LU Decomposition
---

# Week 6: LU Decomposition

## Time Allocation
- **Total Hours**: 12 hours
- **Theory**: 4 hours
- **NumPy Coding**: 4 hours
- **Project**: 4 hours

## Learning Objectives
- Understand LU decomposition and its purpose
- Learn Gaussian elimination systematically
- Implement LU factorization from scratch
- Apply LU decomposition to solve linear systems efficiently

## Theory (4 hours)

### Resources
1. **MIT OCW 18.06**
   - Lecture 4: Factorization into A = LU
   - Lecture 5: Transposes, permutations, spaces R
   - [Watch here](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)

2. **3Blue1Brown - Essence of Linear Algebra**
   - Review Chapter 3: Linear transformations and matrices
   - Understanding matrix decompositions

### Core Concepts

#### LU Decomposition
- **Definition**: A = LU where:
  - L is lower triangular (with 1s on diagonal)
  - U is upper triangular
- **Purpose**: Efficient solving of Ax = b
- **When it exists**: When Gaussian elimination works without row swaps

#### Gaussian Elimination
- Systematic method to convert matrix to upper triangular form
- Elementary row operations:
  - Multiply row by scalar
  - Add multiple of one row to another
  - Swap rows (requires permutation matrix P)

#### Forward and Backward Substitution
- **Forward substitution**: Solve Ly = b (lower triangular)
- **Backward substitution**: Solve Ux = y (upper triangular)
- Both operations are O(n�) vs O(n�) for general systems

#### Permutation Matrices (PA = LU)
- Sometimes need row swaps for numerical stability
- P is permutation matrix
- PA = LU form handles all cases

## NumPy Coding Exercises (4 hours)

### Understanding Triangular Matrices

```python
import numpy as np

# Lower triangular matrix
L = np.array([[1, 0, 0],
              [2, 1, 0],
              [3, 4, 1]])

print("Lower triangular matrix L:")
print(L)
print(f"Diagonal: {np.diag(L)}")

# Upper triangular matrix
U = np.array([[2, 3, 1],
              [0, 4, 2],
              [0, 0, 5]])

print("\nUpper triangular matrix U:")
print(U)
print(f"Diagonal: {np.diag(U)}")

# Check properties
print(f"\nL is lower triangular? {np.allclose(L, np.tril(L))}")
print(f"U is upper triangular? {np.allclose(U, np.triu(U))}")

# Reconstruct original matrix
A = L @ U
print(f"\nOriginal matrix A = L @ U:")
print(A)
```

### Forward Substitution

```python
def forward_substitution(L, b):
    """
    Solve Ly = b where L is lower triangular
    Returns: y
    """
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]

    return y

# Test
L = np.array([[2, 0, 0],
              [4, 3, 0],
              [2, 1, 5]])
b = np.array([6, 19, 23])

y = forward_substitution(L, b)
print("Forward substitution:")
print(f"L:\n{L}")
print(f"b: {b}")
print(f"y: {y}")
print(f"Verification (L @ y): {L @ y}")
print(f"Match? {np.allclose(L @ y, b)}")
```

### Backward Substitution

```python
def backward_substitution(U, y):
    """
    Solve Ux = y where U is upper triangular
    Returns: x
    """
    n = len(y)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]

    return x

# Test
U = np.array([[3, 2, 1],
              [0, 4, 2],
              [0, 0, 5]])
y = np.array([10, 12, 15])

x = backward_substitution(U, y)
print("\nBackward substitution:")
print(f"U:\n{U}")
print(f"y: {y}")
print(f"x: {x}")
print(f"Verification (U @ x): {U @ x}")
print(f"Match? {np.allclose(U @ x, y)}")
```

### LU Decomposition (No Pivoting)

```python
def lu_decomposition(A):
    """
    Perform LU decomposition: A = LU
    No pivoting (assumes no row swaps needed)
    Returns: L, U
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)

    for k in range(n-1):
        # Check for zero pivot
        if abs(U[k, k]) < 1e-10:
            raise ValueError(f"Zero pivot encountered at position {k}")

        # Eliminate column k below diagonal
        for i in range(k+1, n):
            # Compute multiplier
            multiplier = U[i, k] / U[k, k]
            L[i, k] = multiplier

            # Eliminate
            U[i, k:] -= multiplier * U[k, k:]

    return L, U

# Test
A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

print("LU Decomposition (no pivoting):")
print(f"A:\n{A}")

L, U = lu_decomposition(A)

print(f"\nL (lower triangular):\n{L}")
print(f"\nU (upper triangular):\n{U}")

# Verify
A_reconstructed = L @ U
print(f"\nReconstructed A = L @ U:\n{A_reconstructed}")
print(f"Match original? {np.allclose(A, A_reconstructed)}")
```

### LU Decomposition with Partial Pivoting

```python
def lu_decomposition_pivot(A):
    """
    Perform LU decomposition with partial pivoting: PA = LU
    Returns: P, L, U
    """
    n = A.shape[0]
    P = np.eye(n)
    L = np.eye(n)
    U = A.copy().astype(float)

    for k in range(n-1):
        # Find pivot (largest element in column k, from row k downward)
        pivot_row = k + np.argmax(np.abs(U[k:, k]))

        # Swap rows in U, P, and update L
        if pivot_row != k:
            # Swap rows in U
            U[[k, pivot_row], k:] = U[[pivot_row, k], k:]

            # Swap rows in P
            P[[k, pivot_row], :] = P[[pivot_row, k], :]

            # Swap rows in L (only for already computed part)
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # Eliminate column k below diagonal
        for i in range(k+1, n):
            if abs(U[k, k]) < 1e-10:
                continue  # Skip if pivot is zero

            multiplier = U[i, k] / U[k, k]
            L[i, k] = multiplier
            U[i, k:] -= multiplier * U[k, k:]

    return P, L, U

# Test with matrix that needs pivoting
A = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]], dtype=float)

print("\nLU Decomposition with pivoting:")
print(f"A:\n{A}")

P, L, U = lu_decomposition_pivot(A)

print(f"\nP (permutation):\n{P}")
print(f"\nL (lower triangular):\n{L}")
print(f"\nU (upper triangular):\n{U}")

# Verify PA = LU
PA = P @ A
LU = L @ U
print(f"\nPA:\n{PA}")
print(f"LU:\n{LU}")
print(f"PA = LU? {np.allclose(PA, LU)}")
```

### Solving Linear Systems with LU

```python
def solve_lu(A, b):
    """
    Solve Ax = b using LU decomposition
    Steps:
      1. Decompose A = LU
      2. Solve Ly = b (forward substitution)
      3. Solve Ux = y (backward substitution)
    """
    # Get LU decomposition
    L, U = lu_decomposition(A)

    # Forward substitution: Ly = b
    y = forward_substitution(L, b)

    # Backward substitution: Ux = y
    x = backward_substitution(U, y)

    return x

# Test
A = np.array([[3, 2, 1],
              [2, 3, 2],
              [1, 2, 3]], dtype=float)
b = np.array([12, 13, 12], dtype=float)

print("\nSolving Ax = b using LU decomposition:")
print(f"A:\n{A}")
print(f"b: {b}")

x = solve_lu(A, b)
print(f"\nSolution x: {x}")

# Verify
print(f"Verification (A @ x): {A @ x}")
print(f"Matches b? {np.allclose(A @ x, b)}")

# Compare with direct solution
x_direct = np.linalg.solve(A, b)
print(f"Direct solve: {x_direct}")
print(f"Solutions match? {np.allclose(x, x_direct)}")
```

### Using NumPy's Built-in LU

```python
from scipy.linalg import lu

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]], dtype=float)

# SciPy's LU decomposition (with pivoting)
P, L, U = lu(A)

print("Using SciPy's lu():")
print(f"A:\n{A}")
print(f"\nP:\n{P}")
print(f"\nL:\n{L}")
print(f"\nU:\n{U}")

# Verify
print(f"\nP @ L @ U:\n{P @ L @ U}")
print(f"Matches A? {np.allclose(P @ L @ U, A)}")
```

### Efficiency Comparison

```python
import time

# Compare solving methods
sizes = [10, 50, 100, 200]

print("\nEfficiency comparison for solving Ax = b:")
print("Size | LU time | Direct time | Speedup")
print("-" * 50)

for n in sizes:
    A = np.random.randn(n, n)
    b = np.random.randn(n)

    # LU method (decompose once, solve once)
    start = time.time()
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x1 = backward_substitution(U, y)
    lu_time = time.time() - start

    # Direct method
    start = time.time()
    x2 = np.linalg.solve(A, b)
    direct_time = time.time() - start

    speedup = direct_time / lu_time if lu_time > 0 else float('inf')

    print(f"{n:4d} | {lu_time:.6f}s | {direct_time:.6f}s | {speedup:.2f}x")

print("\nNote: NumPy's solve() uses optimized LAPACK routines")
print("Our implementation is for educational purposes")
```

## Project: Solving Multiple Systems Efficiently (4 hours)

When you need to solve Ax = b for multiple different b vectors, LU decomposition is much more efficient.

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearSystemSolver:
    """
    Efficient solver for multiple linear systems with same A
    Uses LU decomposition to avoid repeated factorization
    """

    def __init__(self, A):
        """Initialize with coefficient matrix A"""
        self.A = A.copy()
        self.n = A.shape[0]

        # Perform LU decomposition once
        print("Performing LU decomposition...")
        self.P, self.L, self.U = self._lu_decompose()
        print("LU decomposition complete!")

    def _lu_decompose(self):
        """LU decomposition with partial pivoting"""
        return lu_decomposition_pivot(self.A)

    def solve(self, b):
        """
        Solve Ax = b using stored LU decomposition
        Much faster than solving from scratch each time!
        """
        # Apply permutation to b
        b_permuted = self.P @ b

        # Forward substitution
        y = forward_substitution(self.L, b_permuted)

        # Backward substitution
        x = backward_substitution(self.U, y)

        return x

    def solve_multiple(self, B):
        """
        Solve AX = B where B is a matrix (multiple right-hand sides)
        Each column of B is a different b vector
        Returns matrix X where each column is a solution
        """
        n_systems = B.shape[1]
        X = np.zeros_like(B)

        for i in range(n_systems):
            X[:, i] = self.solve(B[:, i])

        return X

# Example application: Temperature distribution in a 2D grid
class HeatEquationSolver:
    """
    Solve steady-state heat equation using LU decomposition
    Models temperature distribution in a 2D plate
    """

    def __init__(self, n=10):
        """
        n: grid size (n x n interior points)
        """
        self.n = n
        self.N = n * n  # Total interior points

        # Build the system matrix (Laplacian)
        self.A = self._build_laplacian()
        self.solver = LinearSystemSolver(self.A)

    def _build_laplacian(self):
        """
        Build discrete Laplacian matrix for 2D grid
        -4 on diagonal, 1 for neighbors
        """
        N = self.N
        A = np.zeros((N, N))

        for i in range(N):
            A[i, i] = -4

            # Right neighbor
            if (i + 1) % self.n != 0:
                A[i, i + 1] = 1

            # Left neighbor
            if i % self.n != 0:
                A[i, i - 1] = 1

            # Top neighbor
            if i + self.n < N:
                A[i, i + self.n] = 1

            # Bottom neighbor
            if i - self.n >= 0:
                A[i, i - self.n] = 1

        return A

    def solve_heat(self, boundary_conditions):
        """
        Solve for steady-state temperature distribution
        boundary_conditions: dict with 'top', 'bottom', 'left', 'right' temps
        """
        # Build right-hand side vector
        b = np.zeros(self.N)

        bc = boundary_conditions

        for i in range(self.N):
            row = i // self.n
            col = i % self.n

            # Add boundary conditions
            if row == 0:  # Bottom edge
                b[i] -= bc.get('bottom', 0)
            if row == self.n - 1:  # Top edge
                b[i] -= bc.get('top', 0)
            if col == 0:  # Left edge
                b[i] -= bc.get('left', 0)
            if col == self.n - 1:  # Right edge
                b[i] -= bc.get('right', 0)

        # Solve using LU decomposition
        T_flat = self.solver.solve(b)

        # Reshape to 2D grid
        T = T_flat.reshape(self.n, self.n)

        return T

    def visualize(self, T, boundary_conditions):
        """Visualize temperature distribution"""
        # Add boundary values
        n = self.n
        T_full = np.zeros((n + 2, n + 2))
        T_full[1:-1, 1:-1] = T

        # Set boundaries
        bc = boundary_conditions
        T_full[0, :] = bc.get('bottom', 0)
        T_full[-1, :] = bc.get('top', 0)
        T_full[:, 0] = bc.get('left', 0)
        T_full[:, -1] = bc.get('right', 0)

        # Plot
        plt.figure(figsize=(10, 8))
        im = plt.imshow(T_full, cmap='hot', interpolation='bilinear')
        plt.colorbar(im, label='Temperature')
        plt.title('Steady-State Temperature Distribution')
        plt.xlabel('x')
        plt.ylabel('y')

        # Add contour lines
        X, Y = np.meshgrid(range(n + 2), range(n + 2))
        plt.contour(X, Y, T_full, levels=10, colors='black',
                   alpha=0.3, linewidths=0.5)

        plt.show()

# Test: Solve heat equation with different boundary conditions
print("\n" + "="*60)
print("HEAT EQUATION SOLVER")
print("="*60)

heat_solver = HeatEquationSolver(n=20)

# Scenario 1: Hot top, cold bottom
print("\nScenario 1: Hot top (100�), cold sides")
boundary_1 = {'top': 100, 'bottom': 0, 'left': 0, 'right': 0}
T1 = heat_solver.solve_heat(boundary_1)
heat_solver.visualize(T1, boundary_1)

# Scenario 2: Hot sides
print("\nScenario 2: Hot left and right sides")
boundary_2 = {'top': 0, 'bottom': 0, 'left': 100, 'right': 100}
T2 = heat_solver.solve_heat(boundary_2)
heat_solver.visualize(T2, boundary_2)

# Scenario 3: Hot right side only
print("\nScenario 3: Hot right side (100�), cold elsewhere")
boundary_3 = {'top': 0, 'bottom': 0, 'left': 0, 'right': 100}
T3 = heat_solver.solve_heat(boundary_3)
heat_solver.visualize(T3, boundary_3)
```

### Efficiency Analysis

```python
# Compare solving single vs multiple systems
n = 50
A = np.random.randn(n, n)

# Create solver (performs LU once)
solver = LinearSystemSolver(A)

# Solve 100 different systems
n_systems = 100
B = np.random.randn(n, n_systems)

# Method 1: Using LU (efficient)
import time
start = time.time()
X_lu = solver.solve_multiple(B)
time_lu = time.time() - start

# Method 2: Solving each from scratch (inefficient)
start = time.time()
X_direct = np.zeros((n, n_systems))
for i in range(n_systems):
    X_direct[:, i] = np.linalg.solve(A, B[:, i])
time_direct = time.time() - start

print(f"\nSolving {n_systems} systems of size {n}�{n}:")
print(f"LU method: {time_lu:.4f} seconds")
print(f"Direct method: {time_direct:.4f} seconds")
print(f"Speedup: {time_direct/time_lu:.2f}x")
print(f"\nSolutions match? {np.allclose(X_lu, X_direct)}")
```

## Weekly Checkpoint

By the end of Week 6, you should be able to:
-  Understand LU decomposition conceptually
-  Implement forward and backward substitution
-  Perform LU factorization with and without pivoting
-  Solve linear systems efficiently using LU
-  Apply LU decomposition to practical problems
-  Recognize when LU is more efficient than direct methods

## Resources

### Video Lectures
- [MIT 18.06 - Lecture 4: LU Decomposition](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-4-factorization-into-a-lu/)
- [Khan Academy - Gaussian Elimination](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/matrices-elimination/v/matrices-reduced-row-echelon-form-1)

### Reading
- [SciPy LU Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html)
- [Numerical Recipes - LU Decomposition](http://numerical.recipes/)

### Applications
- Solving linear systems in engineering simulations
- Circuit analysis (Kirchhoff's laws)
- Finite element methods
- Computer graphics (lighting calculations)

## Next Week Preview

Week 7 will cover determinants and matrix inverses. You'll learn the geometric meaning of determinants and how to compute and apply matrix inverses. The concepts from LU decomposition will be very useful!