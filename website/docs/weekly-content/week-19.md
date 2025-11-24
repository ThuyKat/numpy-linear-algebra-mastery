---
title: Week 19 - Transformers & Attention
sidebar_label: Week 19 - Transformers & Attention
---

# Week 19: Transformers & Attention Mechanisms

## Time Allocation
**Total**: 14 hours | Theory: 6h | Coding: 4h | Project: 4h

## Learning Objectives
- Understand self-attention mechanism
- Learn multi-head attention mathematics
- Implement transformer components from scratch
- Apply to sequence modeling

## Core Concepts
- **Attention**: Attention(Q,K,V) = softmax(QKᵀ/√d_k)V
- **Multi-Head**: Parallel attention in different subspaces
- **Positional Encoding**: Add position information
- **Layer Normalization**: Normalize across features

## NumPy Coding

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention
    Q, K, V: (batch, seq_len, d_k)
    """
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores + (mask * -1e9)

    # Softmax
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)

    # Apply attention to values
    output = attention_weights @ V

    return output, attention_weights

# Test
batch_size, seq_len, d_k = 2, 5, 8
Q = np.random.randn(batch_size, seq_len, d_k)
K = np.random.randn(batch_size, seq_len, d_k)
V = np.random.randn(batch_size, seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")

# Visualize attention weights for one example
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(weights[0], cmap='viridis')
plt.colorbar(label='Attention Weight')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.title('Attention Weights Heatmap')
plt.show()
```

## Multi-Head Attention

```python
class MultiHeadAttention:
    """Multi-head attention layer"""

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

    def split_heads(self, x):
        """Split into multiple heads"""
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)

    def forward(self, query, key, value, mask=None):
        """Forward pass"""
        batch_size = query.shape[0]

        # Linear projections
        Q = query @ self.W_q
        K = key @ self.W_k
        V = value @ self.W_v

        # Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply attention for each head
        attention_output = np.zeros_like(Q)
        for h in range(self.num_heads):
            attn_out, _ = scaled_dot_product_attention(
                Q[:, h], K[:, h], V[:, h], mask
            )
            attention_output[:, h] = attn_out

        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3)
        attention_output = attention_output.reshape(
            batch_size, -1, self.d_model
        )

        # Final linear
        output = attention_output @ self.W_o

        return output

# Test
mha = MultiHeadAttention(d_model=64, num_heads=8)
X = np.random.randn(2, 10, 64)  # batch=2, seq=10, d_model=64
output = mha.forward(X, X, X)
print(f"Multi-head attention output: {output.shape}")
```

## Positional Encoding

```python
def positional_encoding(seq_len, d_model):
    """
    Create positional encoding matrix
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((seq_len, d_model))

    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe

# Visualize positional encoding
pe = positional_encoding(seq_len=100, d_model=64)

plt.figure(figsize=(12, 8))
plt.imshow(pe.T, aspect='auto', cmap='RdBu')
plt.colorbar()
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding')
plt.show()
```

## Feed-Forward Network

```python
class FeedForward:
    """Position-wise feed-forward network"""

    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2/d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2/d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # Two linear transformations with ReLU
        hidden = np.maximum(0, x @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2
        return output

class TransformerBlock:
    """Single transformer encoder block"""

    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        # Note: Add layer norm in full implementation

    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention.forward(x, x, x)
        x = x + attn_out  # Residual connection

        # Feed-forward with residual
        ff_out = self.feed_forward.forward(x)
        x = x + ff_out  # Residual connection

        return x

# Test
block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
X = np.random.randn(2, 10, 64)
output = block.forward(X)
print(f"Transformer block output: {output.shape}")
```

## Project: Simplified Transformer

```python
class SimpleTransformer:
    """Simplified transformer for sequence-to-sequence"""

    def __init__(self, d_model=64, num_heads=4, num_layers=2, d_ff=256):
        self.d_model = d_model

        # Encoder blocks
        self.encoder_blocks = [
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

    def encode(self, x):
        """Encode input sequence"""
        # Add positional encoding
        seq_len = x.shape[1]
        pe = positional_encoding(seq_len, self.d_model)
        x = x + pe

        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block.forward(x)

        return x

# Example: Sequence encoding
transformer = SimpleTransformer(d_model=64, num_heads=4, num_layers=2)
sequences = np.random.randn(4, 20, 64)  # batch=4, seq=20, d_model=64
encoded = transformer.encode(sequences)
print(f"Encoded sequences: {encoded.shape}")
```

## Resources
- [Attention Is All You Need - Paper](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
