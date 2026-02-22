#!/usr/bin/env python3
"""
Task runner for addition transformer experiments on spot TPU VMs.

Usage:
    python -m infra.spot.addition_task_runner \
        --task-id tiny-2L-32d \
        --output gs://YOUR_GCS_BUCKET/addition-sweep/runs/tiny-2L-32d
"""

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infra.spot.config import ADDITION_SWEEP, GCS_RESULTS_DIR

print = partial(print, flush=True)


# HeartbeatThread removed - orchestrator now handles all state management
# Task runner only logs to wandb, orchestrator polls wandb for completion


def upload_to_gcs(local_path: str, gcs_path: str) -> bool:
    import subprocess
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", local_path, gcs_path],
        capture_output=True, text=True
    )
    return result.returncode == 0


def install_dependencies():
    """Install JAX and dependencies on TPU VM."""
    import subprocess
    print("[setup] Installing dependencies...")
    commands = [
        "python3 -m pip install --upgrade pip",
        "python3 -m pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
        "python3 -m pip install flax optax numpy wandb",
    ]
    for cmd in commands:
        print(f"  $ {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [error] {result.stderr}")
            return False
    return True


@dataclass
class Config:
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    vocab_size: int = 14
    max_seq_len: int = 35
    batch_size: int = 512
    learning_rate: float = 1e-3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    curriculum: Tuple[Tuple[int, int, int], ...] = (
        (1, 3, 2000),
        (1, 6, 5000),
        (1, 10, 20000),
    )
    eval_every: int = 1000
    seed: int = 42


TOKENS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
          '+': 10, '=': 11, '<PAD>': 12, '<EOS>': 13}
PAD_TOKEN, EOS_TOKEN = 12, 13


def preprocess(a: int, b: int) -> Tuple[str, str]:
    return f"{str(a).zfill(10)}+{str(b).zfill(10)}=", str(a + b).zfill(11)[::-1]


def tokenize(s: str) -> List[int]:
    return [TOKENS[c] for c in s]


def run_training(config: Config, output_dir: str, task_id: str = None):
    """Run training and return results dict."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import flax.linen as nn
    from flax.training import train_state

    # Initialize wandb if API key is available
    use_wandb = os.environ.get('WANDB_API_KEY') is not None
    if use_wandb:
        import wandb
        wandb.init(
            project="addition-sweep",
            name=task_id,
            config={
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
                "d_model": config.d_model,
                "d_ff": config.d_ff,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "warmup_steps": config.warmup_steps,
            },
        )
        print(f"[wandb] Initialized: {wandb.run.url}")

    print(f"JAX devices: {jax.devices()}")

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
            attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
            attn = jnp.where(jnp.tril(jnp.ones((T, T))) == 0, -1e9, attn)
            attn = jax.nn.softmax(attn, axis=-1)
            out = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, T, C)
            return nn.Dense(C, use_bias=False)(out)

    class TransformerBlock(nn.Module):
        n_heads: int
        d_model: int
        d_ff: int
        @nn.compact
        def __call__(self, x):
            x = x + CausalSelfAttention(self.n_heads, self.d_model)(nn.LayerNorm()(x))
            h = jax.nn.gelu(nn.Dense(self.d_ff)(nn.LayerNorm()(x)))
            return x + nn.Dense(self.d_model)(h)

    class AdditionTransformer(nn.Module):
        config: Config
        @nn.compact
        def __call__(self, x):
            B, T = x.shape
            cfg = self.config
            x = nn.Embed(cfg.vocab_size, cfg.d_model)(x) + nn.Embed(cfg.max_seq_len, cfg.d_model)(jnp.arange(T))
            for _ in range(cfg.n_layers):
                x = TransformerBlock(cfg.n_heads, cfg.d_model, cfg.d_ff)(x)
            return nn.Dense(cfg.vocab_size, use_bias=False)(nn.LayerNorm()(x))

    # Initialize
    rng = jax.random.PRNGKey(config.seed)
    model = AdditionTransformer(config)
    params = model.init(rng, jnp.ones((1, config.max_seq_len), dtype=jnp.int32))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {n_params:,}")

    total_steps = sum(p[2] for p in config.curriculum)
    lr_schedule = optax.warmup_cosine_decay_schedule(0.0, config.learning_rate, config.warmup_steps, total_steps, config.learning_rate * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(config.grad_clip_norm), optax.adamw(lr_schedule, weight_decay=config.weight_decay))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state, inputs, targets, mask):
        def loss_fn(params):
            logits = state.apply_fn(params, inputs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    def generate_batch(rng, batch_size, min_digits, max_digits, seq_len=35):
        inputs = np.full((batch_size, seq_len), PAD_TOKEN, dtype=np.int32)
        targets = np.full((batch_size, seq_len), PAD_TOKEN, dtype=np.int32)
        mask = np.zeros((batch_size, seq_len), dtype=np.float32)
        for i in range(batch_size):
            n_digits = rng.integers(min_digits, max_digits + 1)
            a, b = int(rng.integers(0, 10**n_digits)), int(rng.integers(0, 10**n_digits))
            inp, tgt = preprocess(a, b)
            inp_tok, tgt_tok = tokenize(inp), tokenize(tgt) + [EOS_TOKEN]
            full_seq, full_tgt = inp_tok + tokenize(tgt), inp_tok[1:] + tgt_tok
            inputs[i, :len(full_seq)] = full_seq
            targets[i, :len(full_tgt)] = full_tgt
            mask[i, len(inp_tok)-1:len(inp_tok)-1+len(tgt_tok)] = 1.0
        return inputs, targets, mask

    def evaluate(state, dataset, batch_size=512):
        correct = 0
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start+batch_size]
            batch_inp = [tokenize(preprocess(a, b)[0]) for a, b, _ in batch]
            padded = np.full((len(batch_inp), max(len(x) for x in batch_inp)), PAD_TOKEN, np.int32)
            for i, t in enumerate(batch_inp): padded[i, :len(t)] = t
            current = jnp.array(padded)
            for _ in range(12):
                logits = state.apply_fn(state.params, current)
                current = jnp.concatenate([current, jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)], axis=1)
            gen = np.array(current[:, padded.shape[1]:])
            for i, (a, b, _) in enumerate(batch):
                digits = [str(t) for t in gen[i] if 0 <= t <= 9 and t != EOS_TOKEN]
                if digits:
                    try:
                        if int(''.join(digits)[::-1]) == a + b: correct += 1
                    except: pass
        return correct / len(dataset)

    # Generate datasets
    print("Generating datasets...")
    rng_val = np.random.default_rng(config.seed + 1000)
    val_data = [(int(rng_val.integers(0, 10**10)), int(rng_val.integers(0, 10**10)), 0) for _ in range(5000)]
    val_data = [(a, b, a+b) for a, b, _ in val_data]
    rng_test = np.random.default_rng(config.seed + 2000)
    test_data = [(int(rng_test.integers(0, 10**10)), int(rng_test.integers(0, 10**10)), 0) for _ in range(10000)]
    test_data = [(a, b, a+b) for a, b, _ in test_data]

    # Training
    np_rng = np.random.default_rng(config.seed)
    global_step, best_val_acc, start_time = 0, 0.0, time.time()
    log = {'config': asdict(config), 'n_params': n_params, 'train_losses': [], 'val_accuracies': []}

    for phase_idx, (min_d, max_d, n_steps) in enumerate(config.curriculum):
        print(f"\n=== Phase {phase_idx+1}: {min_d}-{max_d} digits, {n_steps} steps ===")
        for step in range(n_steps):
            inputs, targets, mask = generate_batch(np_rng, config.batch_size, min_d, max_d)
            state, loss = train_step(state, jnp.array(inputs), jnp.array(targets), jnp.array(mask))
            global_step += 1
            if global_step % 100 == 0:
                print(f"Step {global_step}: loss={float(loss):.6f}, time={time.time()-start_time:.1f}s")
                log['train_losses'].append({'step': global_step, 'loss': float(loss)})
                if use_wandb:
                    wandb.log({"train/loss": float(loss), "step": global_step})
            if global_step % config.eval_every == 0:
                val_acc = evaluate(state, val_data[:1000])
                print(f"  Val acc: {val_acc:.4f}")
                log['val_accuracies'].append({'step': global_step, 'accuracy': val_acc})
                if use_wandb:
                    wandb.log({"val/accuracy": val_acc, "step": global_step})
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    with open(os.path.join(output_dir, 'best_params.pkl'), 'wb') as f:
                        pickle.dump(state.params, f)

    # Final eval
    print("\n=== Final Evaluation ===")
    final_val = evaluate(state, val_data)
    test_acc = evaluate(state, test_data)
    print(f"Val: {final_val:.4f}, Test: {test_acc:.4f}")

    if use_wandb:
        wandb.log({"final/val_accuracy": final_val, "final/test_accuracy": test_acc, "n_params": n_params})
        wandb.finish()

    log.update({'final_val_accuracy': final_val, 'final_test_accuracy': test_acc, 'total_time': time.time() - start_time})
    with open(os.path.join(output_dir, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    return {'n_params': n_params, 'test_accuracy': test_acc, 'val_accuracy': final_val}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--skip-install', action='store_true')
    args = parser.parse_args()

    print(f"[addition_task_runner] Task: {args.task_id}")
    print(f"[addition_task_runner] Output: {args.output}")

    # Find config - supports both old format (5 fields) and new format (7 fields)
    config_entry = next((e for e in ADDITION_SWEEP if e[0] == args.task_id), None)
    if not config_entry:
        print(f"[error] Unknown task_id: {args.task_id}")
        sys.exit(1)

    # Parse config entry (backwards compatible)
    if len(config_entry) == 5:
        _, n_layers, n_heads, d_model, d_ff = config_entry
        lr, warmup_ratio = 1e-3, 0.05
    else:
        _, n_layers, n_heads, d_model, d_ff, lr, warmup_ratio = config_entry

    # Calculate warmup steps as percentage of total curriculum
    total_steps = 27000  # 2000 + 5000 + 20000
    warmup_steps = int(total_steps * warmup_ratio)

    config = Config(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        learning_rate=lr,
        warmup_steps=warmup_steps
    )
    print(f"  Config: {n_layers}L {n_heads}H d={d_model} ff={d_ff} lr={lr} warmup={warmup_steps}")

    # Note: orchestrator handles all GCS state management
    # Task runner only trains and logs to wandb

    try:
        if not args.skip_install:
            if not install_dependencies():
                raise RuntimeError("Dependency install failed")

        output_dir = f"/tmp/addition_{args.task_id}"
        os.makedirs(output_dir, exist_ok=True)

        results = run_training(config, output_dir, task_id=args.task_id)

        # Upload results to GCS
        upload_to_gcs(f"{output_dir}/*", args.output)
        with open(f"/tmp/results_{args.task_id}.json", 'w') as f:
            json.dump(results, f)
        upload_to_gcs(f"/tmp/results_{args.task_id}.json", f"{GCS_RESULTS_DIR}/{args.task_id}.json")

        print(f"[addition_task_runner] Done! test={results['test_accuracy']:.4f}")

    except Exception as e:
        print(f"[error] {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
