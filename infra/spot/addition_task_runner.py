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
    # Architectural options for param reduction
    ffn_bias: bool = True         # Use bias in FFN layers
    tied_embeddings: bool = False # Tie input/output embeddings
    sinusoidal_pos: bool = False  # Use sinusoidal instead of learned positions
    rmsnorm: bool = False         # Use RMSNorm (no bias) instead of LayerNorm
    no_delimiters: bool = False   # Fixed-format input (vocab=10, no +/=/PAD/EOS)
    tied_qk: bool = False         # Share Q and K projections
    ffn_mult: float = 4.0         # FFN expansion factor (default 4x)
    rope: bool = False            # Use Rotary Position Embeddings (RoPE)


# Standard tokenization (vocab=14)
TOKENS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
          '+': 10, '=': 11, '<PAD>': 12, '<EOS>': 13}
PAD_TOKEN, EOS_TOKEN = 12, 13

# No-delimiter tokenization (vocab=10, fixed positions)
# Format: "AAAAAAAAAA BBBBBBBBBB CCCCCCCCCCC" (10+10+11 digits, positions implicit)
TOKENS_NODLM = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
PAD_TOKEN_NODLM, EOS_TOKEN_NODLM = 0, 0  # Pad with zeros in no-delimiter mode


def preprocess(a: int, b: int, no_delimiters: bool = False) -> Tuple[str, str]:
    """Convert integers to model input/output strings.

    Standard mode (no_delimiters=False):
        Input:  "0000000005+0000000007="
        Output: "21000000000"  (12 reversed)

    No-delimiter mode (no_delimiters=True):
        Input:  "00000000050000000007"  (A padded to 10 + B padded to 10)
        Output: "21000000000"  (sum reversed, padded to 11)
    """
    if no_delimiters:
        return f"{str(a).zfill(10)}{str(b).zfill(10)}", str(a + b).zfill(11)[::-1]
    else:
        return f"{str(a).zfill(10)}+{str(b).zfill(10)}=", str(a + b).zfill(11)[::-1]


def tokenize(s: str, no_delimiters: bool = False) -> List[int]:
    """Tokenize string to integer list."""
    tok_map = TOKENS_NODLM if no_delimiters else TOKENS
    return [tok_map[c] for c in s]


def get_sinusoidal_embeddings(seq_len: int, d_model: int):
    """Generate fixed sinusoidal positional embeddings."""
    import numpy as np
    position = np.arange(seq_len)[:, np.newaxis]
    # Handle odd dimensions by using ceil for sin frequencies, floor for cos
    d_sin = (d_model + 1) // 2  # Number of sin channels
    d_cos = d_model // 2        # Number of cos channels
    div_term_sin = np.exp(np.arange(0, d_sin * 2, 2) * (-np.log(10000.0) / d_model))
    div_term_cos = np.exp(np.arange(0, d_cos * 2, 2) * (-np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term_sin)
    if d_cos > 0:
        pe[:, 1::2] = np.cos(position * div_term_cos)
    return pe


def run_training(config: Config, output_dir: str, task_id: str = None):
    """Run training and return results dict."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import flax.linen as nn
    from flax.training import train_state

    # Adjust vocab_size and seq_len for no_delimiters mode
    if config.no_delimiters:
        config.vocab_size = 10  # Only digits 0-9
        config.max_seq_len = 31  # 10 + 10 + 11 = 31

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
                "ffn_bias": config.ffn_bias,
                "tied_embeddings": config.tied_embeddings,
                "sinusoidal_pos": config.sinusoidal_pos,
                "rmsnorm": config.rmsnorm,
                "no_delimiters": config.no_delimiters,
                "tied_qk": config.tied_qk,
            },
        )
        print(f"[wandb] Initialized: {wandb.run.url}")

    print(f"JAX devices: {jax.devices()}")

    # Pre-compute sinusoidal embeddings if needed
    sinusoidal_pe = get_sinusoidal_embeddings(config.max_seq_len, config.d_model) if config.sinusoidal_pos else None

    # Pre-compute RoPE frequencies if needed
    def get_rope_freqs(seq_len, head_dim):
        """Compute RoPE rotation frequencies."""
        # theta_i = 10000^(-2i/d) for i in [0, d/2)
        freqs = 1.0 / (10000.0 ** (jnp.arange(0, head_dim, 2) / head_dim))
        positions = jnp.arange(seq_len)
        # outer product: (seq_len, head_dim/2)
        angles = jnp.outer(positions, freqs)
        return jnp.cos(angles), jnp.sin(angles)

    def apply_rope(x, cos_freqs, sin_freqs):
        """Apply rotary position embeddings to x.
        x: (batch, heads, seq_len, head_dim)
        Returns same shape with rotation applied.
        """
        # Split into pairs for rotation
        x1 = x[..., 0::2]  # even indices
        x2 = x[..., 1::2]  # odd indices
        # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        # cos_freqs, sin_freqs: (seq_len, head_dim/2)
        cos = cos_freqs[None, None, :x.shape[2], :]  # broadcast to (1, 1, seq, dim/2)
        sin = sin_freqs[None, None, :x.shape[2], :]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        # Interleave back
        return jnp.stack([x1_rot, x2_rot], axis=-1).reshape(x.shape)

    rope_cos, rope_sin = None, None
    if config.rope:
        head_dim = config.d_model // config.n_heads
        rope_cos, rope_sin = get_rope_freqs(config.max_seq_len, head_dim)

    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization (no bias, fewer params)."""
        epsilon: float = 1e-6
        @nn.compact
        def __call__(self, x):
            scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
            rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
            return (x / rms) * scale

    class CausalSelfAttention(nn.Module):
        n_heads: int
        d_model: int
        tied_qk: bool = False
        use_rope: bool = False
        @nn.compact
        def __call__(self, x, rope_cos=None, rope_sin=None):
            B, T, C = x.shape
            head_dim = C // self.n_heads

            if self.tied_qk:
                # Tied Q=K: Only learn 2 projections (QK shared, V separate)
                qk = nn.Dense(C, use_bias=False)(x)  # Shared Q and K
                v = nn.Dense(C, use_bias=False)(x)
                q = qk.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
                k = qk.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
                v = v.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            else:
                qkv = nn.Dense(3 * C, use_bias=False)(x)
                q, k, v = jnp.split(qkv, 3, axis=-1)
                q = q.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
                k = k.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
                v = v.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)

            # Apply RoPE if enabled
            if self.use_rope and rope_cos is not None and rope_sin is not None:
                q = apply_rope(q, rope_cos, rope_sin)
                k = apply_rope(k, rope_cos, rope_sin)

            attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
            attn = jnp.where(jnp.tril(jnp.ones((T, T))) == 0, -1e9, attn)
            attn = jax.nn.softmax(attn, axis=-1)
            out = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, T, C)
            return nn.Dense(C, use_bias=False)(out)

    class TransformerBlock(nn.Module):
        n_heads: int
        d_model: int
        d_ff: int
        ffn_bias: bool = True
        rmsnorm: bool = False
        tied_qk: bool = False
        use_rope: bool = False
        @nn.compact
        def __call__(self, x, rope_cos=None, rope_sin=None):
            Norm = RMSNorm if self.rmsnorm else nn.LayerNorm
            x = x + CausalSelfAttention(self.n_heads, self.d_model, self.tied_qk, self.use_rope)(Norm()(x), rope_cos, rope_sin)
            h = jax.nn.gelu(nn.Dense(self.d_ff, use_bias=self.ffn_bias)(Norm()(x)))
            return x + nn.Dense(self.d_model, use_bias=self.ffn_bias)(h)

    class AdditionTransformer(nn.Module):
        config: Config
        @nn.compact
        def __call__(self, x, sinusoidal_pe=None, rope_cos=None, rope_sin=None):
            B, T = x.shape
            cfg = self.config
            Norm = RMSNorm if cfg.rmsnorm else nn.LayerNorm

            tok_emb = nn.Embed(cfg.vocab_size, cfg.d_model)
            x_emb = tok_emb(x)

            # Positional embeddings: learned, sinusoidal, or RoPE (applied in attention)
            if cfg.rope:
                # RoPE: no additive position embedding, rotation applied in attention
                x = x_emb
            elif cfg.sinusoidal_pos and sinusoidal_pe is not None:
                pos_emb = sinusoidal_pe[:T]  # Fixed, no learnable params
                x = x_emb + pos_emb
            else:
                pos_emb = nn.Embed(cfg.max_seq_len, cfg.d_model)(jnp.arange(T))
                x = x_emb + pos_emb

            for _ in range(cfg.n_layers):
                x = TransformerBlock(
                    cfg.n_heads, cfg.d_model, cfg.d_ff,
                    cfg.ffn_bias, cfg.rmsnorm, cfg.tied_qk, cfg.rope
                )(x, rope_cos, rope_sin)
            x = Norm()(x)

            if cfg.tied_embeddings:
                # Reuse token embedding weights for output projection
                return x @ tok_emb.embedding.T
            else:
                return nn.Dense(cfg.vocab_size, use_bias=False)(x)

    # Initialize
    rng = jax.random.PRNGKey(config.seed)
    model = AdditionTransformer(config)
    params = model.init(rng, jnp.ones((1, config.max_seq_len), dtype=jnp.int32),
                       sinusoidal_pe=sinusoidal_pe, rope_cos=rope_cos, rope_sin=rope_sin)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {n_params:,}")

    total_steps = sum(p[2] for p in config.curriculum)
    lr_schedule = optax.warmup_cosine_decay_schedule(0.0, config.learning_rate, config.warmup_steps, total_steps, config.learning_rate * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(config.grad_clip_norm), optax.adamw(lr_schedule, weight_decay=config.weight_decay))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state, inputs, targets, mask):
        def loss_fn(params):
            logits = state.apply_fn(params, inputs, sinusoidal_pe=sinusoidal_pe,
                                   rope_cos=rope_cos, rope_sin=rope_sin)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    # Determine tokens based on mode
    nodlm = config.no_delimiters
    pad_tok = PAD_TOKEN_NODLM if nodlm else PAD_TOKEN
    eos_tok = EOS_TOKEN_NODLM if nodlm else EOS_TOKEN
    seq_len = config.max_seq_len

    def generate_batch(rng, batch_size, min_digits, max_digits):
        inputs = np.full((batch_size, seq_len), pad_tok, dtype=np.int32)
        targets = np.full((batch_size, seq_len), pad_tok, dtype=np.int32)
        mask = np.zeros((batch_size, seq_len), dtype=np.float32)
        for i in range(batch_size):
            n_digits = rng.integers(min_digits, max_digits + 1)
            a, b = int(rng.integers(0, 10**n_digits)), int(rng.integers(0, 10**n_digits))
            inp, tgt = preprocess(a, b, no_delimiters=nodlm)
            inp_tok = tokenize(inp, no_delimiters=nodlm)
            tgt_tok = tokenize(tgt, no_delimiters=nodlm)
            if not nodlm:
                tgt_tok = tgt_tok + [eos_tok]  # Add EOS only in standard mode
            full_seq = inp_tok + tokenize(tgt, no_delimiters=nodlm)
            full_tgt = inp_tok[1:] + tgt_tok
            inputs[i, :len(full_seq)] = full_seq
            targets[i, :len(full_tgt)] = full_tgt
            mask[i, len(inp_tok)-1:len(inp_tok)-1+len(tgt_tok)] = 1.0
        return inputs, targets, mask

    def evaluate(state, dataset, batch_size=512):
        correct = 0
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start+batch_size]
            batch_inp = [tokenize(preprocess(a, b, no_delimiters=nodlm)[0], no_delimiters=nodlm) for a, b, _ in batch]
            padded = np.full((len(batch_inp), max(len(x) for x in batch_inp)), pad_tok, np.int32)
            for i, t in enumerate(batch_inp): padded[i, :len(t)] = t
            current = jnp.array(padded)
            gen_len = 11 if nodlm else 12  # 11 digits for result, +1 for EOS in standard mode
            for _ in range(gen_len):
                logits = state.apply_fn(state.params, current, sinusoidal_pe=sinusoidal_pe,
                                       rope_cos=rope_cos, rope_sin=rope_sin)
                current = jnp.concatenate([current, jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)], axis=1)
            gen = np.array(current[:, padded.shape[1]:])
            for i, (a, b, _) in enumerate(batch):
                digits = [str(t) for t in gen[i] if 0 <= t <= 9]
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
                    wandb.log({"train/loss": float(loss)}, step=global_step)
            if global_step % config.eval_every == 0:
                val_acc = evaluate(state, val_data[:1000])
                print(f"  Val acc: {val_acc:.4f}")
                log['val_accuracies'].append({'step': global_step, 'accuracy': val_acc})
                if use_wandb:
                    wandb.log({"val/accuracy": val_acc}, step=global_step)
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
        wandb.log({"final/val_accuracy": final_val, "final/test_accuracy": test_acc, "n_params": n_params}, step=global_step)
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

    # Find config - supports multiple formats:
    # 5 fields: (task_id, n_layers, n_heads, d_model, d_ff)
    # 7 fields: (task_id, n_layers, n_heads, d_model, d_ff, lr, warmup_ratio)
    # 9 fields: (task_id, n_layers, n_heads, d_model, d_ff, lr, warmup_ratio, ffn_bias, tied_emb)
    # dict format: {"task_id": ..., "n_layers": ..., ...} for full control
    config_entry = next((e for e in ADDITION_SWEEP if (e[0] if isinstance(e, tuple) else e.get("task_id")) == args.task_id), None)
    if not config_entry:
        print(f"[error] Unknown task_id: {args.task_id}")
        sys.exit(1)

    # Parse config entry - support both tuple and dict formats
    if isinstance(config_entry, dict):
        # Dict format - direct mapping
        n_layers = config_entry["n_layers"]
        n_heads = config_entry["n_heads"]
        d_model = config_entry["d_model"]
        d_ff = config_entry["d_ff"]
        lr = config_entry.get("lr", 1e-3)
        warmup_ratio = config_entry.get("warmup", 0.05)
        ffn_bias = config_entry.get("ffn_bias", True)
        tied_emb = config_entry.get("tied_emb", False)
        sinusoidal_pos = config_entry.get("sinusoidal", False)
        rmsnorm = config_entry.get("rmsnorm", False)
        no_delimiters = config_entry.get("no_delim", False)
        tied_qk = config_entry.get("tied_qk", False)
        rope = config_entry.get("rope", False)
    else:
        # Tuple format (backwards compatible)
        lr, warmup_ratio = 1e-3, 0.05
        ffn_bias, tied_emb = True, False
        sinusoidal_pos, rmsnorm, no_delimiters, tied_qk, rope = False, False, False, False, False

        if len(config_entry) == 5:
            _, n_layers, n_heads, d_model, d_ff = config_entry
        elif len(config_entry) == 7:
            _, n_layers, n_heads, d_model, d_ff, lr, warmup_ratio = config_entry
        elif len(config_entry) >= 9:
            _, n_layers, n_heads, d_model, d_ff, lr, warmup_ratio, ffn_bias, tied_emb = config_entry[:9]

    # Calculate warmup steps as percentage of total curriculum
    total_steps = 27000  # 2000 + 5000 + 20000
    warmup_steps = int(total_steps * warmup_ratio)

    config = Config(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        ffn_bias=ffn_bias,
        tied_embeddings=tied_emb,
        sinusoidal_pos=sinusoidal_pos,
        rmsnorm=rmsnorm,
        no_delimiters=no_delimiters,
        tied_qk=tied_qk,
        rope=rope
    )
    flags = []
    if not ffn_bias: flags.append("no-bias")
    if tied_emb: flags.append("tied-emb")
    if sinusoidal_pos: flags.append("sinusoidal")
    if rmsnorm: flags.append("rmsnorm")
    if no_delimiters: flags.append("no-delim")
    if tied_qk: flags.append("tied-qk")
    if rope: flags.append("rope")
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    print(f"  Config: {n_layers}L {n_heads}H d={d_model} ff={d_ff} lr={lr} warmup={warmup_steps}{flag_str}")

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
