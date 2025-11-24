---
title: Week 7 - Determinants & Inverses
sidebar_label: Week 7 - Determinants & Inverses
---

# Week 7: Determinants & Matrix Inverses

## Time Allocation
- **Total Hours**: 12 hours
- **Theory**: 4 hours
- **NumPy Coding**: 4 hours
- **Project**: 4 hours

## Learning Objectives
- Understand determinants and their geometric meaning
- Master matrix inversion and conditions for invertibility
- Learn to solve linear systems using matrix inverses
- Apply these concepts to practical problems

## Theory (4 hours)

### Resources
1. **3Blue1Brown - Essence of Linear Algebra**
   - Chapter 6: The determinant
   - Chapter 7: Inverse matrices, column space and null space
   - [Watch here](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

2. **MIT OCW 18.06**
   - Lecture 18: Properties of determinants
   - Lecture 19: Determinant formulas and cofactors

### Core Concepts

#### Determinants
- Geometric interpretation: scaling factor for area/volume
- Properties: multiplicative, row operations
- Computing: cofactor expansion, row reduction
- det(A) = 0 ⟺ matrix is singular

#### Matrix Inverse
- Definition: A⁻¹A = AA⁻¹ = I
- Conditions: square matrix with det(A) ≠ 0
- Formula: A⁻¹ = (1/det(A)) × adj(A)
- Properties: (AB)⁻¹ = B⁻¹A⁻¹, (Aᵀ)⁻¹ = (A⁻¹)ᵀ

#### Solving Linear Systems
- Ax = b where A is invertible
- Solution: x = A⁻¹b
- Computational considerations

## NumPy Coding Exercises (4 hours)

### Computing Determinants

```python
import numpy as np

# 2x2 determinant
A2 = np.array([[3, 8], [4, 6]])
det_A2 = np.linalg.det(A2)
print(f"2×2 Matrix:\n{A2}")
print(f"Determinant: {det_A2}")

# Manual calculation for 2x2: ad - bc
manual_det = A2[0,0] * A2[1,1] - A2[0,1] * A2[1,0]
print(f"Manual calculation: {manual_det}")
print(f"Match: {np.isclose(det_A2, manual_det)}")

# 3x3 determinant
A3 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 10]])
det_A3 = np.linalg.det(A3)
print(f"\n3×3 Matrix:\n{A3}")
print(f"Determinant: {det_A3}")

# Singular matrix (det = 0)
singular = np.array([[1, 2], [2, 4]])
print(f"\nSingular matrix:\n{singular}")
print(f"Determinant: {np.linalg.det(singular):.10f}")
```

### Geometric Interpretation of Determinants

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def visualize_determinant(matrix):
    """
    Visualize how a matrix transformation scales area
    """
    # Original unit square
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T

    # Transformed square
    transformed = matrix @ unit_square

    # Calculate determinant
    det = np.linalg.det(matrix)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Original square (area = 1)
    ax1.fill(unit_square[0], unit_square[1], alpha=0.3, color='blue')
    ax1.plot(unit_square[0], unit_square[1], 'b-', linewidth=2)
    ax1.set_xlim(-1, 3)
    ax1.set_ylim(-1, 3)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('Original Square\nArea = 1')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)

    # Transformed parallelogram
    ax2.fill(transformed[0], transformed[1], alpha=0.3, color='red')
    ax2.plot(transformed[0], transformed[1], 'r-', linewidth=2)
    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-1, 3)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title(f'Transformed Shape\nArea = |det(A)| = {abs(det):.2f}')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)

    # Draw transformation vectors
    origin = [0, 0]
    ax2.quiver(*origin, *matrix[:, 0], angles='xy', scale_units='xy',
              scale=1, color='green', width=0.01, label='Column 1')
    ax2.quiver(*origin, *matrix[:, 1], angles='xy', scale_units='xy',
              scale=1, color='purple', width=0.01, label='Column 2')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"Matrix:\n{matrix}")
    print(f"Determinant: {det:.4f}")
    print(f"Area scaling factor: {abs(det):.4f}")

# Test with different matrices
matrices = {
    "Scaling (2x)": np.array([[2, 0], [0, 2]]),
    "Shearing": np.array([[1, 1], [0, 1]]),
    "Rotation 45°": np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                              [np.sin(np.pi/4), np.cos(np.pi/4)]]),
    "Reflection": np.array([[1, 0], [0, -1]])
}

for name, matrix in matrices.items():
    print(f"\n{name}:")
    visualize_determinant(matrix)
```

### Properties of Determinants

```python
# Property 1: det(AB) = det(A) × det(B)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

det_A = np.linalg.det(A)
det_B = np.linalg.det(B)
det_AB = np.linalg.det(A @ B)

print("Property: det(AB) = det(A) × det(B)")
print(f"det(A) = {det_A:.4f}")
print(f"det(B) = {det_B:.4f}")
print(f"det(A) × det(B) = {det_A * det_B:.4f}")
print(f"det(AB) = {det_AB:.4f}")
print(f"Equal? {np.isclose(det_A * det_B, det_AB)}")

# Property 2: det(A^T) = det(A)
det_A_transpose = np.linalg.det(A.T)
print(f"\nProperty: det(A^T) = det(A)")
print(f"det(A) = {det_A:.4f}")
print(f"det(A^T) = {det_A_transpose:.4f}")
print(f"Equal? {np.isclose(det_A, det_A_transpose)}")

# Property 3: det(cA) = c^n × det(A) for n×n matrix
c = 2
n = A.shape[0]
det_cA = np.linalg.det(c * A)
expected = (c ** n) * det_A

print(f"\nProperty: det(cA) = c^n × det(A)")
print(f"c = {c}, n = {n}")
print(f"det(cA) = {det_cA:.4f}")
print(f"c^n × det(A) = {expected:.4f}")
print(f"Equal? {np.isclose(det_cA, expected)}")

# Property 4: Swapping rows changes sign
A_swapped = A[[1, 0], :]  # Swap rows
det_swapped = np.linalg.det(A_swapped)
print(f"\nProperty: Swapping rows changes sign")
print(f"det(A) = {det_A:.4f}")
print(f"det(A with swapped rows) = {det_swapped:.4f}")
print(f"Opposite signs? {np.isclose(det_A, -det_swapped)}")
```

### Matrix Inverses

```python
# Computing inverse
A = np.array([[4, 7], [2, 6]])
print(f"Matrix A:\n{A}")

# Check if invertible
det_A = np.linalg.det(A)
print(f"\nDeterminant: {det_A}")
print(f"Invertible? {det_A != 0}")

if det_A != 0:
    # Compute inverse
    A_inv = np.linalg.inv(A)
    print(f"\nInverse A^(-1):\n{A_inv}")

    # Verify: A × A^(-1) = I
    I = A @ A_inv
    print(f"\nA × A^(-1):\n{I}")
    print(f"Is identity? {np.allclose(I, np.eye(2))}")

    # Verify: A^(-1) × A = I
    I2 = A_inv @ A
    print(f"\nA^(-1) × A:\n{I2}")
    print(f"Is identity? {np.allclose(I2, np.eye(2))}")

# Singular matrix (not invertible)
singular = np.array([[1, 2], [2, 4]])
print(f"\n\nSingular matrix:\n{singular}")
print(f"Determinant: {np.linalg.det(singular):.10f}")
try:
    sing_inv = np.linalg.inv(singular)
except np.linalg.LinAlgError as e:
    print(f"Cannot invert: {e}")
```

### Manual 2x2 Inverse Formula

```python
def inverse_2x2(matrix):
    """
    Compute inverse of 2x2 matrix manually
    For [[a, b], [c, d]], inverse is (1/det) × [[d, -b], [-c, a]]
    """
    a, b = matrix[0]
    c, d = matrix[1]

    det = a*d - b*c

    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular (determinant ≈ 0)")

    inv = (1/det) * np.array([[d, -b], [-c, a]])
    return inv

# Test
A = np.array([[4, 7], [2, 6]])
print(f"Matrix A:\n{A}")

# Manual calculation
A_inv_manual = inverse_2x2(A)
print(f"\nManual inverse:\n{A_inv_manual}")

# NumPy calculation
A_inv_numpy = np.linalg.inv(A)
print(f"\nNumPy inverse:\n{A_inv_numpy}")

# Compare
print(f"\nAre they equal? {np.allclose(A_inv_manual, A_inv_numpy)}")
```

### Solving Linear Systems with Inverses

```python
# System: Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([8, 13])

print(f"System: Ax = b")
print(f"A =\n{A}")
print(f"b = {b}")

# Method 1: Using inverse
A_inv = np.linalg.inv(A)
x_inv = A_inv @ b
print(f"\nSolution using A^(-1):")
print(f"x = A^(-1) × b = {x_inv}")

# Verify solution
verification = A @ x_inv
print(f"Verification: A × x = {verification}")
print(f"Matches b? {np.allclose(verification, b)}")

# Method 2: Using np.linalg.solve (more efficient)
x_solve = np.linalg.solve(A, b)
print(f"\nSolution using np.linalg.solve:")
print(f"x = {x_solve}")
print(f"Same result? {np.allclose(x_inv, x_solve)}")
```

### Properties of Matrix Inverses

```python
# Property: (AB)^(-1) = B^(-1) A^(-1)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

AB = A @ B
AB_inv = np.linalg.inv(AB)

# Compute separately
B_inv = np.linalg.inv(B)
A_inv = np.linalg.inv(A)
B_inv_A_inv = B_inv @ A_inv

print("Property: (AB)^(-1) = B^(-1) A^(-1)")
print(f"(AB)^(-1) =\n{AB_inv}")
print(f"\nB^(-1) A^(-1) =\n{B_inv_A_inv}")
print(f"\nEqual? {np.allclose(AB_inv, B_inv_A_inv)}")

# Property: (A^T)^(-1) = (A^(-1))^T
AT_inv = np.linalg.inv(A.T)
A_inv_T = A_inv.T

print(f"\n\nProperty: (A^T)^(-1) = (A^(-1))^T")
print(f"(A^T)^(-1) =\n{AT_inv}")
print(f"\n(A^(-1))^T =\n{A_inv_T}")
print(f"\nEqual? {np.allclose(AT_inv, A_inv_T)}")
```

## Project: Circuit Analysis with Matrix Inverses (4 hours)

Apply matrix operations to analyze electrical circuits using Kirchhoff's laws.

```python
import numpy as np
import matplotlib.pyplot as plt

class CircuitAnalyzer:
    """
    Analyze electrical circuits using matrix methods
    Based on Kirchhoff's Current and Voltage Laws
    """

    def __init__(self):
        self.node_voltages = None
        self.currents = None

    def solve_nodal_analysis(self, conductance_matrix, current_sources):
        """
        Solve circuit using nodal analysis
        G × V = I
        where G is conductance matrix, V is node voltages, I is current sources
        """
        print("Conductance Matrix G:")
        print(conductance_matrix)
        print(f"\nDeterminant: {np.linalg.det(conductance_matrix):.6f}")

        if abs(np.linalg.det(conductance_matrix)) < 1e-10:
            raise ValueError("Singular matrix - cannot solve")

        # Solve for voltages
        self.node_voltages = np.linalg.solve(conductance_matrix, current_sources)

        print(f"\nCurrent Sources I: {current_sources}")
        print(f"\nNode Voltages V: {self.node_voltages}")

        # Verify solution
        verification = conductance_matrix @ self.node_voltages
        print(f"\nVerification (G × V): {verification}")
        print(f"Matches I? {np.allclose(verification, current_sources)}")

        return self.node_voltages

    def calculate_branch_currents(self, node_voltages, resistances):
        """Calculate currents through branches using Ohm's law"""
        currents = {}
        for (node1, node2), R in resistances.items():
            V1 = node_voltages[node1] if node1 >= 0 else 0  # Ground = 0V
            V2 = node_voltages[node2] if node2 >= 0 else 0
            I = (V1 - V2) / R
            currents[f"I_{node1}_{node2}"] = I
            print(f"Current from node {node1} to {node2}: {I:.4f} A")

        return currents

    def power_analysis(self, node_voltages, resistances):
        """Calculate power dissipation in each resistor"""
        total_power = 0
        print("\nPower Analysis:")

        for (node1, node2), R in resistances.items():
            V1 = node_voltages[node1] if node1 >= 0 else 0
            V2 = node_voltages[node2] if node2 >= 0 else 0
            V_drop = V1 - V2
            P = (V_drop ** 2) / R
            total_power += P
            print(f"Resistor R_{node1}_{node2} ({R}Ω): {P:.4f} W")

        print(f"\nTotal Power Dissipated: {total_power:.4f} W")
        return total_power

# Example: Simple circuit with 3 nodes
# Node 0 is ground (reference)
# Current source at node 1: 2A
# Resistors: R01=2Ω, R12=3Ω, R02=6Ω

"""
Circuit Diagram:
    2A →  Node 1
          /  \
       2Ω     3Ω
        /       \
    Node 0    Node 2
    (GND)       |
          \   6Ω |
           \_____|
"""

# Build conductance matrix for nodes 1 and 2
# G[i,j] = -1/R_ij for i≠j (off-diagonal)
# G[i,i] = sum of conductances connected to node i (diagonal)

R01 = 2  # Between node 0 and 1
R12 = 3  # Between node 1 and 2
R02 = 6  # Between node 0 and 2

# Conductance matrix for nodes 1 and 2
G = np.array([
    [1/R01 + 1/R12, -1/R12],        # Node 1
    [-1/R12, 1/R12 + 1/R02]         # Node 2
])

# Current sources (positive = current entering node)
I = np.array([2, 0])  # 2A into node 1, 0A into node 2

# Solve circuit
analyzer = CircuitAnalyzer()
print("=" * 60)
print("CIRCUIT ANALYSIS")
print("=" * 60)
voltages = analyzer.solve_nodal_analysis(G, I)

# Calculate branch currents
resistances = {
    (1, 0): R01,
    (1, 2): R12,
    (2, 0): R02
}

print("\n" + "=" * 60)
print("BRANCH CURRENTS")
print("=" * 60)
currents = analyzer.calculate_branch_currents(voltages, resistances)

# Power analysis
print("\n" + "=" * 60)
analyzer.power_analysis(voltages, resistances)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Node voltages
nodes = ['Node 1', 'Node 2']
ax1.bar(nodes, voltages, color=['blue', 'red'])
ax1.set_ylabel('Voltage (V)')
ax1.set_title('Node Voltages')
ax1.grid(True, alpha=0.3)

# Branch currents
branches = list(currents.keys())
current_values = list(currents.values())
ax2.bar(branches, current_values, color='green')
ax2.set_ylabel('Current (A)')
ax2.set_title('Branch Currents')
ax2.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

### Extension: More Complex Circuit

```python
# 4-node circuit (3 nodes + ground)
"""
    5A →  Node 1 ─── 2Ω ─── Node 2
           |                   |
          4Ω                  3Ω
           |                   |
       Node 3 ───── 5Ω ─────  Node 0 (GND)
"""

R12 = 2
R13 = 4
R23 = 3
R30 = 5

# Conductance matrix (nodes 1, 2, 3)
G_complex = np.array([
    [1/R12 + 1/R13, -1/R12, -1/R13],         # Node 1
    [-1/R12, 1/R12 + 1/R23, -1/R23],         # Node 2
    [-1/R13, -1/R23, 1/R13 + 1/R23 + 1/R30]  # Node 3
])

I_complex = np.array([5, 0, 0])  # 5A into node 1

print("\n\n" + "=" * 60)
print("COMPLEX CIRCUIT ANALYSIS (4 nodes)")
print("=" * 60)

analyzer2 = CircuitAnalyzer()
voltages_complex = analyzer2.solve_nodal_analysis(G_complex, I_complex)

resistances_complex = {
    (1, 2): R12,
    (1, 3): R13,
    (2, 3): R23,
    (3, 0): R30
}

print("\n" + "=" * 60)
print("BRANCH CURRENTS")
print("=" * 60)
analyzer2.calculate_branch_currents(voltages_complex, resistances_complex)
analyzer2.power_analysis(voltages_complex, resistances_complex)
```

## Weekly Checkpoint

By the end of Week 7, you should be able to:
- ✅ Calculate and interpret determinants
- ✅ Understand geometric meaning of determinants (area/volume scaling)
- ✅ Compute matrix inverses and check invertibility
- ✅ Solve linear systems using matrix inverses
- ✅ Apply these concepts to real-world problems (circuits, etc.)

## Resources

### Video Lectures
- [3Blue1Brown - The Determinant](https://www.youtube.com/watch?v=Ip3X9LOh2dk)
- [3Blue1Brown - Inverse matrices](https://www.youtube.com/watch?v=uQhTuRlWMxw)
- [MIT 18.06 - Lecture 18](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)

### Reading
- [Determinants - Khan Academy](https://www.khanacademy.org/math/linear-algebra/matrix-transformations/determinant-depth/v/linear-algebra-determinant-when-row-is-added)
- [Matrix Inverse - Wikipedia](https://en.wikipedia.org/wiki/Invertible_matrix)

## Next Week Preview
Week 8 will cover eigenvalues and eigenvectors - one of the most important concepts in linear algebra with applications throughout machine learning.
