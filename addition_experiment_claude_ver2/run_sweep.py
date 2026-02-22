#!/usr/bin/env python3
"""
Sweep script for addition transformer - finds minimal model size.
Runs multiple configurations and reports results.
"""

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from functools import partial
from typing import Dict, List, Tuple, Any

# Force unbuffered output
print = partial(print, flush=True)

print("Starting sweep script...", flush=True)
print("Importing JAX...", flush=True)

import jax
import jax.numpy as jnp

print(f"JAX imported. Devices: {jax.devices()}", flush=True)

import numpy as np
import optax
import flax.linen as nn
from flax.training import train_state

print("All imports complete.", flush=True)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Model - these will be varied in sweep
    n_layers: int = 2
    n_heads: int = 2
    d_model: int = 64
    d_ff: int = 256
    vocab_size: int = 14
    max_seq_len: int = 35
    dropout_rate: float = 0.0

    # Training
    batch_size: int = 512
    learning_rate: float = 1e-3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    # Curriculum: (min_digits, max_digits, steps)
    curriculum: Tuple[Tuple[int, int, int], ...] = (
        (1, 3, 2000),
        (1, 6, 5000),
        (1, 10, 20000),
    )

    # Evaluation
    eval_every: int = 1000
    val_size: int = 5000
    test_size: int = 10000

    seed: int = 42


# Sweep configurations: (n_layers, n_heads, d_model, d_ff, name)
SWEEP_CONFIGS = [
    # Tiny models (~50-100K params)
    (2, 2, 32, 128, "tiny_2L_32d"),
    (2, 2, 48, 192, "tiny_2L_48d"),
    (2, 2, 64, 256, "small_2L_64d"),

    # Small models (~100-200K params)
    (2, 4, 64, 256, "small_2L_64d_4h"),
    (3, 2, 64, 256, "small_3L_64d"),
    (2, 2, 96, 384, "small_2L_96d"),

    # Medium models (~200-400K params)
    (3, 4, 64, 256, "med_3L_64d_4h"),
    (2, 4, 128, 512, "med_2L_128d"),
    (4, 2, 64, 256, "med_4L_64d"),

    # Reference (known to work)
    (4, 4, 128, 512, "large_4L_128d"),
]


# Token definitions
TOKENS = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '+': 10, '=': 11, '<PAD>': 12, '<EOS>': 13
}
TOKENS_INV = {v: k for k, v in TOKENS.items()}
PAD_TOKEN = 12
EOS_TOKEN = 13


# =============================================================================
# Data Pipeline
# =============================================================================

def preprocess(a: int, b: int) -> Tuple[str, str]:
    a_str = str(a).zfill(10)
    b_str = str(b).zfill(10)
    c = a + b
    c_str = str(c).zfill(11)
    c_reversed = c_str[::-1]
    return f"{a_str}+{b_str}=", c_reversed


def tokenize(s: str) -> List[int]:
    return [TOKENS[c] for c in s]


def generate_batch(rng: np.random.Generator, batch_size: int, min_digits: int, max_digits: int, seq_len: int = 35):
    inputs = np.full((batch_size, seq_len), PAD_TOKEN, dtype=np.int32)
    targets = np.full((batch_size, seq_len), PAD_TOKEN, dtype=np.int32)
    mask = np.zeros((batch_size, seq_len), dtype=np.float32)

    for i in range(batch_size):
        n_digits = rng.integers(min_digits, max_digits + 1)
        max_val = 10 ** n_digits
        a, b = int(rng.integers(0, max_val)), int(rng.integers(0, max_val))

        input_str, target_str = preprocess(a, b)
        input_tokens = tokenize(input_str)
        target_tokens = tokenize(target_str) + [EOS_TOKEN]

        full_seq = input_tokens + tokenize(target_str)
        full_target = input_tokens[1:] + target_tokens

        inputs[i, :len(full_seq)] = full_seq
        targets[i, :len(full_target)] = full_target

        eq_pos = len(input_tokens) - 1
        mask[i, eq_pos:eq_pos + len(target_tokens)] = 1.0

    return inputs, targets, mask


def generate_fixed_dataset(seed: int, size: int) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    max_val = 10 ** 10
    return [(int(rng.integers(0, max_val)), int(rng.integers(0, max_val)), 0)
            for _ in range(size)]


# =============================================================================
# Model
# =============================================================================

class CausalSelfAttention(nn.Module):
    n_heads: int
    d_model: int

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        head_dim = C // self.n_heads

        qkv = nn.Dense(3 * C, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / jnp.sqrt(head_dim)
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        mask = jnp.tril(jnp.ones((T, T)))
        attn = jnp.where(mask == 0, -1e9, attn)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return nn.Dense(C, use_bias=False)(out)


class TransformerBlock(nn.Module):
    n_heads: int
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x):
        x = x + CausalSelfAttention(self.n_heads, self.d_model)(nn.LayerNorm()(x))
        h = nn.Dense(self.d_ff)(nn.LayerNorm()(x))
        h = jax.nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        return x + h


class AdditionTransformer(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        B, T = x.shape
        cfg = self.config

        tok_emb = nn.Embed(cfg.vocab_size, cfg.d_model)(x)
        pos_emb = nn.Embed(cfg.max_seq_len, cfg.d_model)(jnp.arange(T))
        x = tok_emb + pos_emb

        for _ in range(cfg.n_layers):
            x = TransformerBlock(cfg.n_heads, cfg.d_model, cfg.d_ff)(x)

        x = nn.LayerNorm()(x)
        return nn.Dense(cfg.vocab_size, use_bias=False)(x)


def count_params(params) -> int:
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


# =============================================================================
# Training
# =============================================================================

def create_train_state(rng, config: Config):
    model = AdditionTransformer(config)
    dummy = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy)

    total_steps = sum(p[2] for p in config.curriculum)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=total_steps,
        end_value=config.learning_rate * 0.1
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adamw(lr_schedule, weight_decay=config.weight_decay)
    )

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(jax.jit, donate_argnums=(0,))
def train_step(state, inputs, targets, mask):
    def loss_fn(params):
        logits = state.apply_fn(params, inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def evaluate_accuracy(state, dataset: List[Tuple[int, int, int]], batch_size: int = 512) -> float:
    correct = 0

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]

        batch_inputs = []
        for a, b, _ in batch:
            input_str, _ = preprocess(a, b)
            batch_inputs.append(tokenize(input_str))

        max_len = max(len(x) for x in batch_inputs)
        padded = np.full((len(batch_inputs), max_len), PAD_TOKEN, dtype=np.int32)
        for i, toks in enumerate(batch_inputs):
            padded[i, :len(toks)] = toks

        current = jnp.array(padded)
        for _ in range(12):
            logits = state.apply_fn(state.params, current)
            next_tok = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            current = jnp.concatenate([current, next_tok], axis=1)

        gen = np.array(current[:, max_len:])
        for i, (a, b, _) in enumerate(batch):
            pred_digits = []
            for t in gen[i]:
                if t == EOS_TOKEN:
                    break
                if 0 <= t <= 9:
                    pred_digits.append(str(t))
            if pred_digits:
                try:
                    pred = int(''.join(pred_digits)[::-1])
                    if pred == a + b:
                        correct += 1
                except:
                    pass

    return correct / len(dataset)


def train_single_config(config: Config, config_name: str, output_dir: str, val_data, test_data):
    """Train a single configuration and return results."""
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"  Layers: {config.n_layers}, Heads: {config.n_heads}, d_model: {config.d_model}, d_ff: {config.d_ff}")

    rng = jax.random.PRNGKey(config.seed)
    state = create_train_state(rng, config)
    n_params = count_params(state.params)
    print(f"  Parameters: {n_params:,}")

    np_rng = np.random.default_rng(config.seed)
    global_step = 0
    best_val_acc = 0.0
    start_time = time.time()

    train_losses = []
    val_accs = []

    for phase_idx, (min_d, max_d, n_steps) in enumerate(config.curriculum):
        print(f"  Phase {phase_idx + 1}: {min_d}-{max_d} digits, {n_steps} steps")

        for step in range(n_steps):
            inputs, targets, mask = generate_batch(np_rng, config.batch_size, min_d, max_d)
            state, loss = train_step(state, jnp.array(inputs), jnp.array(targets), jnp.array(mask))
            global_step += 1

            if global_step % 500 == 0:
                train_losses.append({'step': global_step, 'loss': float(loss)})

            if global_step % config.eval_every == 0:
                val_acc = evaluate_accuracy(state, val_data[:1000])
                val_accs.append({'step': global_step, 'accuracy': val_acc})
                print(f"    Step {global_step}: loss={float(loss):.6f}, val_acc={val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                # Early stopping if we hit 100%
                if val_acc >= 0.999:
                    print(f"    Early stopping - 99.9%+ achieved!")
                    break

        if best_val_acc >= 0.999:
            break

    # Final evaluation
    final_val_acc = evaluate_accuracy(state, val_data)
    test_acc = evaluate_accuracy(state, test_data)
    total_time = time.time() - start_time

    print(f"  Final: val_acc={final_val_acc:.4f}, test_acc={test_acc:.4f}, time={total_time:.1f}s")

    result = {
        'config_name': config_name,
        'n_layers': config.n_layers,
        'n_heads': config.n_heads,
        'd_model': config.d_model,
        'd_ff': config.d_ff,
        'n_params': n_params,
        'final_val_accuracy': final_val_acc,
        'final_test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'total_time': total_time,
        'total_steps': global_step,
        'train_losses': train_losses,
        'val_accuracies': val_accs,
    }

    # Save individual result
    with open(os.path.join(output_dir, f'{config_name}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def run_sweep(output_dir: str, config_idx: int = None):
    """Run sweep of all configurations."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate shared val/test sets
    print("Generating validation set...")
    val_data = generate_fixed_dataset(42 + 1000, 5000)
    for i, (a, b, _) in enumerate(val_data):
        val_data[i] = (a, b, a + b)

    print("Generating test set...")
    test_data = generate_fixed_dataset(42 + 2000, 10000)
    for i, (a, b, _) in enumerate(test_data):
        test_data[i] = (a, b, a + b)

    # Select configs to run
    if config_idx is not None:
        configs_to_run = [SWEEP_CONFIGS[config_idx]]
    else:
        configs_to_run = SWEEP_CONFIGS

    results = []
    for n_layers, n_heads, d_model, d_ff, name in configs_to_run:
        config = Config(
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
        )
        result = train_single_config(config, name, output_dir, val_data, test_data)
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)
    print(f"{'Config':<25} {'Params':>10} {'Test Acc':>10} {'Time':>10}")
    print("-"*60)

    # Sort by params (ascending)
    results.sort(key=lambda x: x['n_params'])

    for r in results:
        status = "✓" if r['final_test_accuracy'] >= 0.99 else "✗"
        print(f"{r['config_name']:<25} {r['n_params']:>10,} {r['final_test_accuracy']:>9.4f} {r['total_time']:>9.1f}s {status}")

    # Find smallest model with >=99% accuracy
    successful = [r for r in results if r['final_test_accuracy'] >= 0.99]
    if successful:
        best = min(successful, key=lambda x: x['n_params'])
        print(f"\n*** BEST: {best['config_name']} with {best['n_params']:,} params, {best['final_test_accuracy']:.4f} test accuracy ***")
    else:
        print("\n*** No configuration achieved >=99% accuracy ***")

    # Save summary
    with open(os.path.join(output_dir, 'sweep_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./sweep_output', help='Output directory')
    parser.add_argument('--config', type=int, default=None, help='Run single config by index')
    args = parser.parse_args()

    run_sweep(args.output, args.config)