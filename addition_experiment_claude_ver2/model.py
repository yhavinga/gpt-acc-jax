"""Minimal transformer model in Flax for addition task."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional
import numpy as np


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    n_heads: int
    d_model: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        B, T, C = x.shape
        head_dim = C // self.n_heads

        # QKV projection
        qkv = nn.Dense(3 * C, use_bias=False, name='qkv_proj')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        q = q.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / jnp.sqrt(head_dim)
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Causal mask
        mask = jnp.tril(jnp.ones((T, T)))
        attn = jnp.where(mask == 0, -1e9, attn)

        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=not train)

        # Apply attention to values
        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        out = nn.Dense(C, use_bias=False, name='out_proj')(out)
        out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=not train)

        return out


class MLP(nn.Module):
    """Feed-forward network."""
    d_model: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Dense(self.d_ff, name='fc1')(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.d_model, name='fc2')(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    n_heads: int
    d_model: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Pre-norm attention
        residual = x
        x = nn.LayerNorm(name='ln1')(x)
        x = CausalSelfAttention(
            n_heads=self.n_heads,
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name='attn'
        )(x, train=train)
        x = residual + x

        # Pre-norm MLP
        residual = x
        x = nn.LayerNorm(name='ln2')(x)
        x = MLP(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
            name='mlp'
        )(x, train=train)
        x = residual + x

        return x


class AdditionTransformer(nn.Module):
    """
    Minimal decoder-only transformer for addition.

    Architecture:
    - Token embeddings + learned positional embeddings
    - N transformer blocks (pre-norm)
    - Final layer norm + linear head
    """
    vocab_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        B, T = x.shape

        # Token embeddings
        tok_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model, name='tok_emb')(x)

        # Positional embeddings (learned)
        pos = jnp.arange(T)
        pos_emb = nn.Embed(num_embeddings=self.max_seq_len, features=self.d_model, name='pos_emb')(pos)

        x = tok_emb + pos_emb
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Transformer blocks
        for i in range(self.n_layers):
            x = TransformerBlock(
                n_heads=self.n_heads,
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name=f'block_{i}'
            )(x, train=train)

        # Final layer norm
        x = nn.LayerNorm(name='ln_f')(x)

        # Output projection to vocabulary
        logits = nn.Dense(self.vocab_size, use_bias=False, name='lm_head')(x)

        return logits


def create_model(config) -> AdditionTransformer:
    """Create model from config."""
    return AdditionTransformer(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        dropout_rate=config.dropout_rate
    )


def count_parameters(params) -> int:
    """Count total number of parameters."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


if __name__ == "__main__":
    # Test the model
    from config import ModelConfig

    config = ModelConfig()
    model = create_model(config)

    # Initialize
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 32), dtype=jnp.int32)

    params = model.init(rng, dummy_input, train=False)

    # Count parameters
    n_params = count_parameters(params)
    print(f"Total parameters: {n_params:,}")

    # Test forward pass
    logits = model.apply(params, dummy_input, train=False)
    print(f"Output shape: {logits.shape}")
