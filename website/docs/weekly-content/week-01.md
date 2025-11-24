---
title: Week 1 - Vectors & NumPy Basics
sidebar_label: Week 1 - Vectors & NumPy Basics
---

# Week 1: Vectors & NumPy Basics

## Time Allocation
- **Total Hours**: 12 hours
- **Theory**: 4 hours
- **NumPy Coding**: 4 hours
- **Project**: 4 hours

## Learning Objectives
- Understand vectors as mathematical objects
- Master NumPy array creation and manipulation
- Learn vector operations (addition, scalar multiplication)
- Apply vector concepts to real problems

## Theory (4 hours)

### Resources
1. **3Blue1Brown - Essence of Linear Algebra**
   - Chapter 1: Vectors, what even are they?
   - [Watch here](https://www.youtube.com/watch?v=fNk_zzaMoSs)

2. **Khan Academy**
   - Introduction to vectors
   - Vector addition and scalar multiplication

### Core Concepts

#### What is a Vector?
- **Physicist's View**: Arrows with magnitude and direction
- **Computer Scientist's View**: Ordered list of numbers
- **Mathematician's View**: Element of a vector space
- **ML Perspective**: Feature representation

#### Vector Notation
- Column vector: v = [v₁, v₂, ..., vₙ]ᵀ
- Row vector: v = [v₁, v₂, ..., vₙ]
- Components/entries/coordinates

#### Vector Operations
- **Addition**: u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]
- **Scalar Multiplication**: cv = [cv₁, cv₂, ..., cvₙ]
- **Properties**: Commutative, associative, distributive

#### Geometric Interpretation
- Vectors as points in space
- Vector addition as "tip-to-tail"
- Scalar multiplication as stretching/shrinking

## NumPy Coding Exercises (4 hours)

### Creating NumPy Arrays

```python
import numpy as np

# Create vectors (1D arrays)
v1 = np.array([1, 2, 3])
print(f"Vector v1: {v1}")
print(f"Type: {type(v1)}")
print(f"Shape: {v1.shape}")
print(f"Dimension: {v1.ndim}")
print(f"Data type: {v1.dtype}")

# Create from different data types
v2 = np.array([1.0, 2.5, 3.7])  # Float array
v3 = np.array([1, 2, 3], dtype=np.float64)  # Explicit dtype

# Special arrays
zeros = np.zeros(5)
ones = np.ones(5)
arange = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # 5 evenly spaced points from 0 to 1

print(f"\nZeros: {zeros}")
print(f"Ones: {ones}")
print(f"Arange: {arange}")
print(f"Linspace: {linspace}")

# Random vectors
random_vector = np.random.rand(5)  # Uniform [0, 1)
normal_vector = np.random.randn(5)  # Standard normal

print(f"\nRandom (uniform): {random_vector}")
print(f"Random (normal): {normal_vector}")
```

### Vector Operations

```python
# Vector addition
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

w = u + v
print(f"u + v = {w}")

# Scalar multiplication
c = 3
scaled = c * u
print(f"{c} * u = {scaled}")

# Element-wise operations
product = u * v  # Element-wise multiplication (NOT dot product!)
print(f"u * v (element-wise) = {product}")

# Vector subtraction
diff = v - u
print(f"v - u = {diff}")

# Combining operations
result = 2*u + 3*v
print(f"2u + 3v = {result}")
```

### Array Indexing and Slicing

```python
v = np.array([10, 20, 30, 40, 50])

# Indexing (0-based)
print(f"First element: {v[0]}")
print(f"Last element: {v[-1]}")
print(f"Third element: {v[2]}")

# Slicing [start:stop:step]
print(f"First three: {v[:3]}")  # [10, 20, 30]
print(f"Last two: {v[-2:]}")    # [40, 50]
print(f"Every other: {v[::2]}") # [10, 30, 50]
print(f"Reversed: {v[::-1]}")   # [50, 40, 30, 20, 10]

# Modifying elements
v[0] = 100
print(f"After modification: {v}")

# Boolean indexing
mask = v > 30
print(f"Mask (v > 30): {mask}")
print(f"Values > 30: {v[mask]}")
```

### Mathematical Functions

```python
v = np.array([1, 4, 9, 16, 25])

# Universal functions (ufuncs)
sqrt_v = np.sqrt(v)
exp_v = np.exp(v)
log_v = np.log(v)

print(f"Original: {v}")
print(f"Square root: {sqrt_v}")
print(f"Exponential: {exp_v}")
print(f"Natural log: {log_v}")

# Trigonometric functions
angles = np.array([0, np.pi/4, np.pi/2, np.pi])
sin_vals = np.sin(angles)
cos_vals = np.cos(angles)

print(f"\nAngles: {angles}")
print(f"Sin: {sin_vals}")
print(f"Cos: {cos_vals}")

# Aggregate functions
data = np.array([2, 4, 6, 8, 10])
print(f"\nData: {data}")
print(f"Sum: {np.sum(data)}")
print(f"Mean: {np.mean(data)}")
print(f"Std: {np.std(data)}")
print(f"Min: {np.min(data)}")
print(f"Max: {np.max(data)}")
```

### Vector Magnitude (Norm)

```python
def magnitude(v):
    """Calculate magnitude (L2 norm) of a vector"""
    return np.sqrt(np.sum(v**2))

# Or use NumPy's built-in
v = np.array([3, 4])
mag1 = magnitude(v)
mag2 = np.linalg.norm(v)

print(f"Vector: {v}")
print(f"Magnitude (manual): {mag1}")
print(f"Magnitude (numpy): {mag2}")

# Unit vector (normalized)
def normalize(v):
    """Create unit vector in same direction"""
    return v / np.linalg.norm(v)

unit_v = normalize(v)
print(f"Unit vector: {unit_v}")
print(f"Magnitude of unit vector: {np.linalg.norm(unit_v)}")
```

### Visualizing Vectors

```python
import matplotlib.pyplot as plt

def plot_vectors(vectors, colors=None, labels=None):
    """Plot 2D vectors"""
    plt.figure(figsize=(8, 8))

    if colors is None:
        colors = ['blue'] * len(vectors)
    if labels is None:
        labels = [f'v{i}' for i in range(len(vectors))]

    for vec, color, label in zip(vectors, colors, labels):
        plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy',
                   scale=1, color=color, width=0.01, label=label)

    # Set axis properties
    all_coords = np.array(vectors)
    max_val = np.max(np.abs(all_coords)) + 1
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Vector Visualization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

# Example: Vector addition visualization
u = np.array([3, 2])
v = np.array([1, 3])
w = u + v

plot_vectors([u, v, w],
            colors=['blue', 'red', 'green'],
            labels=['u', 'v', 'u+v'])
```

## Project: Vector-Based Recommendation System (4 hours)

Build a simple recommendation system using vector similarity.

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleRecommender:
    """
    Movie recommender using vector similarity
    Each movie is represented as a vector of features
    """

    def __init__(self):
        # Movie database: [action, comedy, drama, romance, sci-fi]
        self.movies = {
            'Terminator': np.array([5, 1, 2, 0, 5]),
            'The Notebook': np.array([0, 1, 5, 5, 0]),
            'Superbad': np.array([1, 5, 2, 1, 0]),
            'Inception': np.array([4, 1, 3, 1, 5]),
            'Bridesmaids': np.array([1, 5, 2, 2, 0]),
            'Interstellar': np.array([3, 1, 4, 1, 5]),
            'The Matrix': np.array([5, 1, 2, 0, 5]),
            'When Harry Met Sally': np.array([0, 3, 3, 5, 0]),
        }

        self.genre_names = ['Action', 'Comedy', 'Drama', 'Romance', 'Sci-Fi']

    def cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(v1, v2)
        magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)

        if magnitude_product == 0:
            return 0

        return dot_product / magnitude_product

    def recommend(self, liked_movie, n=3):
        """
        Recommend movies similar to the liked movie
        Returns top n recommendations
        """
        if liked_movie not in self.movies:
            return "Movie not found!"

        target_vector = self.movies[liked_movie]
        similarities = {}

        for movie_name, movie_vector in self.movies.items():
            if movie_name != liked_movie:
                sim = self.cosine_similarity(target_vector, movie_vector)
                similarities[movie_name] = sim

        # Sort by similarity (descending)
        sorted_movies = sorted(similarities.items(),
                              key=lambda x: x[1],
                              reverse=True)

        return sorted_movies[:n]

    def visualize_movies(self, movie_names=None):
        """Visualize movies in 2D using first two features"""
        if movie_names is None:
            movie_names = list(self.movies.keys())

        plt.figure(figsize=(12, 8))

        for movie_name in movie_names:
            vec = self.movies[movie_name]
            plt.scatter(vec[0], vec[1], s=200, alpha=0.6)
            plt.annotate(movie_name, (vec[0], vec[1]),
                        fontsize=9, ha='center')

        plt.xlabel(f'{self.genre_names[0]} Score')
        plt.ylabel(f'{self.genre_names[1]} Score')
        plt.title('Movies in Feature Space (Action vs Comedy)')
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.5, 5.5)
        plt.ylim(-0.5, 5.5)
        plt.show()

    def visualize_recommendation(self, liked_movie):
        """Visualize the recommendation in vector space"""
        if liked_movie not in self.movies:
            return "Movie not found!"

        recommendations = self.recommend(liked_movie, n=3)

        # Create radar chart for genre comparison
        angles = np.linspace(0, 2 * np.pi, len(self.genre_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Plot liked movie
        values = self.movies[liked_movie].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'{liked_movie} (Liked)', color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')

        # Plot recommendations
        colors = ['red', 'green', 'orange']
        for i, (rec_movie, similarity) in enumerate(recommendations):
            values = self.movies[rec_movie].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2,
                   label=f'{rec_movie} (sim: {similarity:.2f})',
                   color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        # Formatting
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.genre_names)
        ax.set_ylim(0, 5)
        ax.set_title(f'Recommendations based on "{liked_movie}"',
                    size=15, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.show()

# Test the recommender
recommender = SimpleRecommender()

# Get recommendations
print("If you liked 'Inception', you might also like:")
recommendations = recommender.recommend('Inception', n=3)
for movie, similarity in recommendations:
    print(f"  {movie} (similarity: {similarity:.3f})")

print("\nIf you liked 'The Notebook', you might also like:")
recommendations = recommender.recommend('The Notebook', n=3)
for movie, similarity in recommendations:
    print(f"  {movie} (similarity: {similarity:.3f})")

# Visualizations
recommender.visualize_movies()
recommender.visualize_recommendation('Inception')
```

### Extension: User Profile

```python
class UserProfile:
    """Create user profile based on watched movies"""

    def __init__(self, recommender):
        self.recommender = recommender
        self.profile = None

    def create_profile(self, watched_movies, ratings):
        """
        Create user profile vector as weighted average of watched movies
        watched_movies: list of movie names
        ratings: list of ratings (1-5) for each movie
        """
        vectors = []
        weights = []

        for movie, rating in zip(watched_movies, ratings):
            if movie in self.recommender.movies:
                vectors.append(self.recommender.movies[movie])
                weights.append(rating)

        if not vectors:
            return None

        # Weighted average
        vectors = np.array(vectors)
        weights = np.array(weights).reshape(-1, 1)
        self.profile = np.average(vectors, axis=0, weights=weights.flatten())

        return self.profile

    def recommend_for_user(self, n=5):
        """Recommend movies based on user profile"""
        if self.profile is None:
            return "Create profile first!"

        similarities = {}
        for movie_name, movie_vector in self.recommender.movies.items():
            sim = self.recommender.cosine_similarity(self.profile, movie_vector)
            similarities[movie_name] = sim

        sorted_movies = sorted(similarities.items(),
                              key=lambda x: x[1],
                              reverse=True)

        return sorted_movies[:n]

# Test user profile
user = UserProfile(recommender)

# User has watched and rated these movies
watched = ['Inception', 'Interstellar', 'The Matrix']
ratings = [5, 5, 4]

profile = user.create_profile(watched, ratings)
print(f"\nUser profile vector: {profile}")

print("\nTop recommendations for this user:")
recommendations = user.recommend_for_user(n=5)
for movie, similarity in recommendations:
    print(f"  {movie} (similarity: {similarity:.3f})")
```

## Weekly Checkpoint

By the end of Week 1, you should be able to:
- ✅ Create and manipulate NumPy arrays
- ✅ Perform vector operations (addition, scalar multiplication)
- ✅ Index and slice arrays efficiently
- ✅ Calculate vector magnitude and create unit vectors
- ✅ Visualize vectors in 2D space
- ✅ Apply vectors to real-world problems (recommendations)

## Resources

### Video Lectures
- [3Blue1Brown - Vectors](https://www.youtube.com/watch?v=fNk_zzaMoSs)
- [Khan Academy - Vector Intro](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/vectors/v/vector-introduction-linear-algebra)

### Reading
- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy Array Basics](https://numpy.org/doc/stable/user/basics.html)

### Practice
- [NumPy Exercises](https://github.com/rougier/numpy-100)
- [Project Euler](https://projecteuler.net/) (use NumPy for solutions)

## Next Week Preview

Week 2 will introduce matrices and linear transformations. You'll learn how matrices can transform vectors and visualize these transformations geometrically. Make sure you're comfortable with vector operations before moving forward!