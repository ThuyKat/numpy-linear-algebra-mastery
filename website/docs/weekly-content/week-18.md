---
title: Week 18 - Recurrent Neural Networks
sidebar_label: Week 18 - RNNs
---

# Week 18: Recurrent Neural Networks (RNNs)

## Time Allocation
**Total**: 13 hours | Theory: 5h | Coding: 4h | Project: 4h

## Learning Objectives
- Understand sequential data processing
- Learn RNN, LSTM, and GRU mathematics
- Implement backpropagation through time (BPTT)
- Apply to time series and text

## Core Concepts
- **RNN**: `h_t = tanh(W_hh h_{t-1} + W_xh x_t + b)`
- **LSTM**: Forget, input, and output gates
- **BPTT**: Backpropagate through time steps
- **Vanishing Gradients**: Challenge with long sequences

## NumPy Coding

```python
import numpy as np

class VanillaRNN:
    """Simple RNN cell"""

    def __init__(self, input_dim, hidden_dim):
        # Initialize weights
        self.Wxh = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bh = np.zeros(hidden_dim)

    def forward(self, x, h_prev):
        """Single time step forward"""
        h = np.tanh(x @ self.Wxh + h_prev @ self.Whh + self.bh)
        return h

    def forward_sequence(self, X):
        """Forward through entire sequence"""
        batch_size, seq_len, input_dim = X.shape
        hidden_dim = self.Whh.shape[0]

        # Initialize hidden state
        h = np.zeros((batch_size, hidden_dim))
        h_history = []

        for t in range(seq_len):
            h = self.forward(X[:, t, :], h)
            h_history.append(h)

        return np.array(h_history).transpose(1, 0, 2)  # (batch, seq, hidden)

# Test
rnn = VanillaRNN(input_dim=10, hidden_dim=20)
X = np.random.randn(5, 8, 10)  # batch=5, seq_len=8, input_dim=10
output = rnn.forward_sequence(X)
print(f"RNN output shape: {output.shape}")
```

## LSTM Implementation

```python
class LSTM:
    """LSTM cell"""

    def __init__(self, input_dim, hidden_dim):
        # Gates: forget, input, cell, output
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.Wf = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wi = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wc = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wo = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale

        self.bf = np.zeros(hidden_dim)
        self.bi = np.zeros(hidden_dim)
        self.bc = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x, h_prev, c_prev):
        """LSTM forward pass"""
        # Concatenate input and previous hidden
        combined = np.concatenate([x, h_prev], axis=1)

        # Gates
        f = self.sigmoid(combined @ self.Wf + self.bf)  # Forget gate
        i = self.sigmoid(combined @ self.Wi + self.bi)  # Input gate
        c_tilde = np.tanh(combined @ self.Wc + self.bc) # Candidate cell
        o = self.sigmoid(combined @ self.Wo + self.bo)  # Output gate

        # Update cell and hidden state
        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)

        return h, c

    def forward_sequence(self, X):
        """Forward through sequence"""
        batch_size, seq_len, input_dim = X.shape
        hidden_dim = self.Wf.shape[1]

        h = np.zeros((batch_size, hidden_dim))
        c = np.zeros((batch_size, hidden_dim))
        h_history = []

        for t in range(seq_len):
            h, c = self.forward(X[:, t, :], h, c)
            h_history.append(h)

        return np.array(h_history).transpose(1, 0, 2)

# Test
lstm = LSTM(input_dim=10, hidden_dim=20)
output = lstm.forward_sequence(X)
print(f"LSTM output shape: {output.shape}")
```

## Project: Text Generation

```python
class CharRNN:
    """Character-level RNN for text generation"""

    def __init__(self, vocab_size, hidden_dim=128):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # RNN parameters
        self.Wxh = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.Why = np.random.randn(hidden_dim, vocab_size) * 0.01
        self.bh = np.zeros(hidden_dim)
        self.by = np.zeros(vocab_size)

    def forward(self, x, h):
        """Forward pass"""
        h = np.tanh(x @ self.Wxh + h @ self.Whh + self.bh)
        y = h @ self.Why + self.by
        # Softmax
        exp_y = np.exp(y - np.max(y))
        p = exp_y / np.sum(exp_y)
        return p, h

    def sample(self, seed_char, length=100):
        """Generate text"""
        h = np.zeros(self.hidden_dim)
        indices = [seed_char]

        for _ in range(length):
            x = np.zeros(self.vocab_size)
            x[indices[-1]] = 1

            p, h = self.forward(x, h)
            idx = np.random.choice(range(self.vocab_size), p=p)
            indices.append(idx)

        return indices

# Example usage
vocab_size = 26  # a-z
model = CharRNN(vocab_size, hidden_dim=64)

# Generate text starting from 'a' (index 0)
generated = model.sample(seed_char=0, length=50)
print(f"Generated sequence: {generated[:20]}...")
```

## Resources
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
