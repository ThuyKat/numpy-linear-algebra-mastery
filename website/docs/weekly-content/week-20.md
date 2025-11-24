---
title: Week 20 - Capstone Project & Integration
sidebar_label: Week 20 - Capstone Project
---

# Week 20: Capstone Project & Integration

## Time Allocation
**Total**: 20 hours
- **Planning**: 4 hours
- **Implementation**: 12 hours
- **Documentation**: 4 hours

## Learning Objectives
- Integrate all learned concepts
- Build end-to-end ML system
- Apply best practices
- Document and present work

## Capstone Project Options

Choose ONE of the following projects that interests you most:

### Option 1: Image Classification System

Build a complete image classification pipeline from scratch.

**Components**:
- Data preprocessing and augmentation
- CNN architecture implementation
- Training with various optimizers
- Evaluation and visualization
- Model compression using SVD

```python
class ImageClassificationSystem:
    """
    Complete image classification system
    Integrates: Week 9 (SVD), Week 16 (NNs), Week 17 (CNNs)
    """

    def __init__(self):
        self.model = None
        self.history = {'loss': [], 'accuracy': []}

    def preprocess_data(self, X, y):
        """Normalize and augment data"""
        # Normalize to [0, 1]
        X = X.astype('float32') / 255.0

        # Center data
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-7
        X = (X - mean) / std

        return X, y

    def build_model(self):
        """Build CNN architecture"""
        from week17 import Conv2D, MaxPool2D
        from week16 import Dense, ReLU, Softmax

        self.model = [
            Conv2D(1, 32, kernel_size=3),
            ReLU(),
            MaxPool2D(pool_size=2),
            Conv2D(32, 64, kernel_size=3),
            ReLU(),
            MaxPool2D(pool_size=2),
            # Flatten and dense layers
            Dense(64 * 5 * 5, 128),
            ReLU(),
            Dense(128, 10),
            Softmax()
        ]

    def train(self, X_train, y_train, X_val, y_val,
              epochs=50, batch_size=32):
        """Training loop with validation"""
        best_val_acc = 0

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self._train_epoch(
                X_train, y_train, batch_size
            )

            # Validation
            val_loss, val_acc = self._evaluate(X_val, y_val)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model('best_model.npz')

            # Log
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    def compress_model(self, k=50):
        """Compress model using SVD (Week 9)"""
        for layer in self.model:
            if hasattr(layer, 'W'):
                U, s, Vt = np.linalg.svd(layer.W, full_matrices=False)
                layer.W = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    def visualize_results(self):
        """Plot training history and confusion matrix"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training curves
        axes[0].plot(self.history['loss'], label='Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Training Loss')

        axes[1].plot(self.history['accuracy'], label='Train Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Training Accuracy')

        plt.tight_layout()
        plt.show()

# Usage example
# system = ImageClassificationSystem()
# system.build_model()
# system.train(X_train, y_train, X_val, y_val)
# system.compress_model(k=50)
# system.visualize_results()
```

### Option 2: NLP Sentiment Analysis

Build a sentiment classification system using transformers.

**Components**:
- Text preprocessing and tokenization
- Word embeddings using PCA (Week 10)
- Transformer encoder (Week 19)
- Classification head
- Attention visualization

```python
class SentimentAnalysisSystem:
    """
    Sentiment analysis with transformers
    Integrates: Week 10 (PCA), Week 19 (Transformers)
    """

    def __init__(self, vocab_size, d_model=128, num_heads=4):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding layer
        self.embeddings = np.random.randn(vocab_size, d_model) * 0.01

        # Transformer encoder
        from week19 import TransformerBlock
        self.transformer = TransformerBlock(d_model, num_heads, d_ff=512)

        # Classification head
        from week16 import Dense, Softmax
        self.classifier = [
            Dense(d_model, 2),  # Binary classification
            Softmax()
        ]

    def preprocess_text(self, texts, max_len=50):
        """Tokenize and pad sequences"""
        # Simple character-level tokenization
        sequences = []
        for text in texts:
            seq = [ord(c) % self.vocab_size for c in text[:max_len]]
            seq += [0] * (max_len - len(seq))  # Pad
            sequences.append(seq)
        return np.array(sequences)

    def forward(self, sequences):
        """Forward pass"""
        # Embed tokens
        embedded = self.embeddings[sequences]

        # Transformer encoding
        encoded = self.transformer.forward(embedded)

        # Pool (mean over sequence)
        pooled = np.mean(encoded, axis=1)

        # Classify
        logits = pooled
        for layer in self.classifier:
            logits = layer.forward(logits)

        return logits

# Usage
# system = SentimentAnalysisSystem(vocab_size=256)
# texts = ["I love this!", "This is terrible"]
# sequences = system.preprocess_text(texts)
# predictions = system.forward(sequences)
```

### Option 3: Recommender System

Build a matrix factorization-based recommender.

**Components**:
- SVD for collaborative filtering (Week 9)
- Gradient descent optimization (Week 12)
- PCA for dimensionality reduction (Week 10)
- Evaluation metrics

```python
class RecommenderSystem:
    """
    Matrix factorization recommender
    Integrates: Week 9 (SVD), Week 10 (PCA), Week 12 (Optimization)
    """

    def __init__(self, n_users, n_items, k=50):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k

        # Initialize user and item embeddings
        self.P = np.random.randn(n_users, k) * 0.01  # User factors
        self.Q = np.random.randn(n_items, k) * 0.01  # Item factors

    def fit_svd(self, ratings_matrix):
        """Fit using SVD"""
        # Replace missing values with means
        filled = ratings_matrix.copy()
        row_means = np.nanmean(filled, axis=1, keepdims=True)
        filled[np.isnan(filled)] = row_means[np.isnan(filled)]

        # SVD
        U, s, Vt = np.linalg.svd(filled, full_matrices=False)

        # Keep top k components
        self.P = U[:, :self.k] @ np.diag(np.sqrt(s[:self.k]))
        self.Q = (np.diag(np.sqrt(s[:self.k])) @ Vt[:self.k, :]).T

    def fit_gd(self, user_ids, item_ids, ratings,
               epochs=100, lr=0.01, reg=0.01):
        """Fit using gradient descent"""
        for epoch in range(epochs):
            total_loss = 0

            for u, i, r in zip(user_ids, item_ids, ratings):
                # Prediction
                pred = self.P[u] @ self.Q[i]
                error = r - pred

                # Gradients
                dP = -error * self.Q[i] + reg * self.P[u]
                dQ = -error * self.P[u] + reg * self.Q[i]

                # Update
                self.P[u] -= lr * dP
                self.Q[i] -= lr * dQ

                total_loss += error ** 2

            if epoch % 10 == 0:
                rmse = np.sqrt(total_loss / len(ratings))
                print(f"Epoch {epoch}, RMSE: {rmse:.4f}")

    def predict(self, user_id, item_id):
        """Predict rating"""
        return self.P[user_id] @ self.Q[item_id]

    def recommend(self, user_id, top_k=10):
        """Get top-k recommendations for user"""
        scores = self.P[user_id] @ self.Q.T
        top_items = np.argsort(scores)[-top_k:][::-1]
        return top_items

# Usage
# system = RecommenderSystem(n_users=1000, n_items=500, k=50)
# system.fit_gd(user_ids, item_ids, ratings)
# recommendations = system.recommend(user_id=42, top_k=10)
```

## Project Requirements

### Code Quality
- ✅ Well-documented functions
- ✅ Modular, reusable code
- ✅ Proper error handling
- ✅ Unit tests for key functions

### Experimentation
- ✅ Multiple hyperparameter configurations
- ✅ Comparison of different approaches
- ✅ Ablation studies

### Documentation
- ✅ README with project overview
- ✅ Architecture diagrams
- ✅ Results and analysis
- ✅ Future improvements

### Presentation
- ✅ Problem statement
- ✅ Approach and methodology
- ✅ Results visualization
- ✅ Lessons learned

## Integration Checklist

Review all 20 weeks and identify which concepts to integrate:

- **Week 1-5**: Foundations (vectors, matrices, norms)
- **Week 6**: LU Decomposition
- **Week 7**: Determinants, inverses
- **Week 8**: Eigenvalues
- **Week 9**: SVD
- **Week 10**: PCA
- **Week 11**: QR decomposition
- **Week 12**: Optimization
- **Week 13**: Linear systems
- **Week 14**: Matrix calculus
- **Week 15**: Advanced decompositions
- **Week 16**: Neural networks
- **Week 17**: CNNs
- **Week 18**: RNNs
- **Week 19**: Transformers

## Final Deliverables

1. **Code Repository**
   - All implementation files
   - Tests and examples
   - Requirements.txt

2. **Documentation**
   - Technical report (5-10 pages)
   - Code documentation
   - Usage examples

3. **Presentation**
   - 10-minute presentation
   - Slides with visualizations
   - Demo (if applicable)

4. **Reflection**
   - What worked well
   - Challenges encountered
   - What you'd do differently
   - Next steps in your learning journey

## Congratulations! 🎉

You've completed a comprehensive 20-week journey through NumPy and Linear Algebra for Machine Learning!

### What You've Accomplished

- ✅ Mastered NumPy fundamentals
- ✅ Deep understanding of linear algebra concepts
- ✅ Implemented ML algorithms from scratch
- ✅ Built neural networks, CNNs, RNNs, Transformers
- ✅ Applied mathematics to real-world problems

### Next Steps

1. **Deepen Your Knowledge**
   - Advanced optimization techniques
   - Probabilistic graphical models
   - Reinforcement learning mathematics

2. **Apply to Real Projects**
   - Kaggle competitions
   - Open source contributions
   - Research papers

3. **Stay Current**
   - Follow latest ML research
   - Experiment with new architectures
   - Continue learning in public

## Resources for Continued Learning

- **Books**: "Deep Learning" (Goodfellow), "Pattern Recognition and Machine Learning" (Bishop)
- **Courses**: Stanford CS229, Fast.ai, DeepLearning.AI
- **Communities**: Reddit r/MachineLearning, Twitter #MLTwitter
- **Papers**: arXiv.org, Papers with Code

---

**Thank you for following this learning journey!** Keep building, keep learning, and share your knowledge with others. 🚀
