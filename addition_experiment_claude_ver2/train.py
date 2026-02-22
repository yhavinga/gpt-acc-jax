"""Training loop for addition transformer."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax.training import train_state
import json
import time
import os
from typing import Dict, Any, Tuple, Optional
from functools import partial
import pickle

from config import ExperimentConfig, PAD_TOKEN, EOS_TOKEN, TOKENS
from model import create_model, count_parameters
from data import (
    generate_batch, generate_fixed_dataset, prepare_batch_from_dataset,
    preprocess, tokenize, detokenize, postprocess
)


class TrainState(train_state.TrainState):
    """Custom train state with dropout RNG."""
    dropout_rng: jax.random.PRNGKey


def create_train_state(rng: jax.random.PRNGKey, config: ExperimentConfig) -> TrainState:
    """Initialize model and optimizer."""
    model = create_model(config.model)

    # Initialize model
    init_rng, dropout_rng = jax.random.split(rng)
    dummy_input = jnp.ones((1, config.model.max_seq_len), dtype=jnp.int32)
    params = model.init(init_rng, dummy_input, train=False)

    # Create optimizer with warmup + cosine decay
    total_steps = sum(phase[2] for phase in config.training.curriculum_phases)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=total_steps,
        end_value=config.training.learning_rate * 0.1
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.training.grad_clip_norm),
        optax.adamw(learning_rate=lr_schedule, weight_decay=config.training.weight_decay)
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=dropout_rng
    )


@partial(jax.jit, static_argnums=(3,))
def train_step(
    state: TrainState,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray
) -> Tuple[TrainState, Dict[str, float]]:
    """Single training step."""
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        logits = state.apply_fn(params, inputs, train=True, rngs={'dropout': dropout_rng})
        # Cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        # Mask out padding
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)

    return state, {'loss': loss}


@jax.jit
def eval_step(
    state: TrainState,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray
) -> Dict[str, float]:
    """Evaluation step (no gradient)."""
    logits = state.apply_fn(state.params, inputs, train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)

    # Token accuracy on masked positions
    preds = jnp.argmax(logits, axis=-1)
    correct = ((preds == targets) * mask).sum()
    total = mask.sum()
    token_acc = correct / (total + 1e-8)

    return {'loss': loss, 'token_acc': token_acc}


def generate_autoregressive(
    state: TrainState,
    input_tokens: jnp.ndarray,
    max_new_tokens: int = 12,
    temperature: float = 0.0
) -> jnp.ndarray:
    """Generate tokens autoregressively."""
    # input_tokens shape: (batch, seq_len)
    current = input_tokens

    for _ in range(max_new_tokens):
        # Get logits for last position
        logits = state.apply_fn(state.params, current, train=False)
        next_logits = logits[:, -1, :]

        if temperature == 0.0:
            # Greedy
            next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
        else:
            # Sample
            next_token = jax.random.categorical(
                jax.random.PRNGKey(0), next_logits / temperature
            ).reshape(-1, 1)

        current = jnp.concatenate([current, next_token], axis=1)

        # Stop if all sequences have generated EOS
        if jnp.all(next_token == EOS_TOKEN):
            break

    return current


def evaluate_exact_match(
    state: TrainState,
    dataset: list,
    batch_size: int = 256
) -> Tuple[float, list]:
    """
    Evaluate exact-match accuracy on a dataset.

    Returns:
        (accuracy, failure_cases)
    """
    correct = 0
    total = 0
    failures = []

    # Process in batches
    n_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(dataset))
        batch_data = dataset[start:end]

        # Prepare input tokens (just the input part, up to and including '=')
        batch_inputs = []
        for a, b, _ in batch_data:
            input_str, _ = preprocess(a, b, max_digits=10)
            input_tokens = tokenize(input_str)
            batch_inputs.append(input_tokens)

        # Pad to same length
        max_len = max(len(x) for x in batch_inputs)
        padded_inputs = np.full((len(batch_inputs), max_len), PAD_TOKEN, dtype=np.int32)
        for i, tokens in enumerate(batch_inputs):
            padded_inputs[i, :len(tokens)] = tokens

        # Generate autoregressively
        generated = generate_autoregressive(
            state,
            jnp.array(padded_inputs),
            max_new_tokens=12  # 11 digits + EOS
        )

        # Check predictions
        for i, (a, b, c) in enumerate(batch_data):
            gen_tokens = generated[i, max_len:].tolist()

            # Extract digits until EOS or end
            pred_digits = []
            for t in gen_tokens:
                if t == EOS_TOKEN:
                    break
                if 0 <= t <= 9:
                    pred_digits.append(str(t))

            # Postprocess: reverse to get actual number
            if pred_digits:
                pred_str = ''.join(pred_digits)
                try:
                    pred = int(pred_str[::-1])
                except ValueError:
                    pred = -1
            else:
                pred = -1

            if pred == c:
                correct += 1
            else:
                if len(failures) < 100:  # Keep first 100 failures
                    failures.append({
                        'a': a, 'b': b, 'expected': c, 'predicted': pred,
                        'pred_str': ''.join(pred_digits) if pred_digits else ''
                    })

            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, failures


def train(config: ExperimentConfig, exp_dir: str):
    """Main training loop."""
    print(f"Starting training: {config.experiment_name}")
    print(f"Experiment directory: {exp_dir}")

    # Initialize
    rng = jax.random.PRNGKey(config.seed)
    state = create_train_state(rng, config)

    n_params = count_parameters(state.params)
    print(f"Model parameters: {n_params:,}")

    # Generate fixed validation and test sets
    print("Generating validation set...")
    val_dataset = generate_fixed_dataset(
        seed=config.seed + 1000,
        size=config.training.val_size,
        min_digits=10,
        max_digits=10
    )

    print("Generating test set...")
    test_dataset = generate_fixed_dataset(
        seed=config.seed + 2000,
        size=config.training.test_size,
        min_digits=10,
        max_digits=10
    )

    # Training log
    log = {
        'config': {
            'model': vars(config.model),
            'training': {k: v for k, v in vars(config.training).items()
                        if not k.startswith('_')},
            'seed': config.seed
        },
        'n_params': n_params,
        'train_losses': [],
        'val_accuracies': [],
        'steps': []
    }

    # Training loop
    np_rng = np.random.default_rng(config.seed)
    global_step = 0
    best_val_acc = 0.0
    start_time = time.time()

    for phase_idx, (min_digits, max_digits, n_steps) in enumerate(config.training.curriculum_phases):
        print(f"\n=== Phase {phase_idx + 1}: digits {min_digits}-{max_digits}, {n_steps} steps ===")

        for step in range(n_steps):
            # Generate batch
            inputs, targets, mask = generate_batch(
                np_rng,
                config.training.batch_size,
                min_digits=min_digits,
                max_digits=max_digits,
                seq_len=config.model.max_seq_len
            )

            # Training step
            state, metrics = train_step(
                state,
                jnp.array(inputs),
                jnp.array(targets),
                jnp.array(mask)
            )

            global_step += 1

            # Logging
            if global_step % config.training.log_every == 0:
                loss = float(metrics['loss'])
                elapsed = time.time() - start_time
                print(f"Step {global_step}: loss={loss:.4f}, time={elapsed:.1f}s")
                log['train_losses'].append({'step': global_step, 'loss': loss})

            # Evaluation
            if global_step % config.training.eval_every == 0:
                val_acc, _ = evaluate_exact_match(state, val_dataset[:1000])  # Subset for speed
                print(f"  Validation accuracy: {val_acc:.4f}")
                log['val_accuracies'].append({'step': global_step, 'accuracy': val_acc})
                log['steps'].append(global_step)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(state, os.path.join(exp_dir, 'checkpoints', 'best.pkl'))
                    print(f"  New best! Saved checkpoint.")

            # Periodic save
            if global_step % config.training.save_every == 0:
                save_checkpoint(state, os.path.join(exp_dir, 'checkpoints', f'step_{global_step}.pkl'))

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_val_acc, val_failures = evaluate_exact_match(state, val_dataset)
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    test_acc, test_failures = evaluate_exact_match(state, test_dataset)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save final results
    log['final_val_accuracy'] = final_val_acc
    log['final_test_accuracy'] = test_acc
    log['total_time'] = time.time() - start_time

    with open(os.path.join(exp_dir, 'logs', 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    # Save failure cases
    with open(os.path.join(exp_dir, 'failure_cases.json'), 'w') as f:
        json.dump({
            'val_failures': val_failures[:50],
            'test_failures': test_failures[:50]
        }, f, indent=2)

    # Save final checkpoint
    save_checkpoint(state, os.path.join(exp_dir, 'checkpoints', 'final.pkl'))

    return state, log


def save_checkpoint(state: TrainState, path: str):
    """Save checkpoint to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({
            'params': state.params,
            'opt_state': state.opt_state,
            'step': state.step
        }, f)


def load_checkpoint(path: str, state: TrainState) -> TrainState:
    """Load checkpoint from file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return state.replace(
        params=data['params'],
        opt_state=data['opt_state'],
        step=data['step']
    )


if __name__ == "__main__":
    config = ExperimentConfig()
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    train(config, exp_dir)
