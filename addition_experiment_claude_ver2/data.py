"""Data generation and preprocessing for addition task."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List
from config import TOKENS, TOKENS_INV, PAD_TOKEN, EOS_TOKEN


def preprocess(a: int, b: int, max_digits: int = 10) -> Tuple[str, str]:
    """
    Convert two integers to model input and target strings.

    Format:
    - Input: "{a}+{b}=" with zero-padding to max_digits
    - Output: reversed sum with EOS (reversed to match carry propagation order)

    Args:
        a: First operand (0 to 10^10 - 1)
        b: Second operand (0 to 10^10 - 1)
        max_digits: Maximum digits per operand

    Returns:
        (input_str, target_str) tuple
    """
    # Zero-pad operands to max_digits
    a_str = str(a).zfill(max_digits)
    b_str = str(b).zfill(max_digits)

    # Compute sum and reverse it (to align with carry propagation)
    c = a + b
    c_str = str(c).zfill(max_digits + 1)  # Sum can have one more digit
    c_reversed = c_str[::-1]

    input_str = f"{a_str}+{b_str}="
    target_str = c_reversed

    return input_str, target_str


def postprocess(output_str: str) -> int:
    """
    Convert model output back to integer.

    Args:
        output_str: Reversed digit string from model

    Returns:
        The integer sum
    """
    # Remove EOS token if present
    output_str = output_str.replace('<EOS>', '').strip()
    # Remove any non-digit characters
    output_str = ''.join(c for c in output_str if c.isdigit())
    if not output_str:
        return 0
    # Reverse to get actual number
    return int(output_str[::-1])


def tokenize(s: str) -> List[int]:
    """Convert string to token IDs."""
    return [TOKENS[c] for c in s]


def detokenize(tokens: List[int]) -> str:
    """Convert token IDs back to string."""
    return ''.join(TOKENS_INV[t] for t in tokens if t not in [PAD_TOKEN])


def create_example(a: int, b: int, max_digits: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a single training example.

    Returns:
        (input_tokens, target_tokens) as numpy arrays
    """
    input_str, target_str = preprocess(a, b, max_digits)

    # Full sequence: input + target + EOS
    full_input = input_str + target_str
    full_target = target_str + '<EOS>'

    input_tokens = tokenize(full_input)
    target_tokens = tokenize(input_str) + tokenize(target_str + '<EOS>')

    # For causal LM: input is [input + target], labels are shifted
    # We'll use teacher forcing: predict next token at each position

    return np.array(input_tokens), np.array(target_tokens)


def generate_batch(
    rng: np.random.Generator,
    batch_size: int,
    min_digits: int = 1,
    max_digits: int = 10,
    seq_len: int = 35
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a batch of training examples.

    Args:
        rng: NumPy random generator
        batch_size: Number of examples
        min_digits: Minimum digits per operand
        max_digits: Maximum digits per operand
        seq_len: Sequence length for padding

    Returns:
        (inputs, targets, mask) arrays of shape (batch_size, seq_len)
    """
    inputs = np.full((batch_size, seq_len), PAD_TOKEN, dtype=np.int32)
    targets = np.full((batch_size, seq_len), PAD_TOKEN, dtype=np.int32)
    mask = np.zeros((batch_size, seq_len), dtype=np.float32)

    for i in range(batch_size):
        # Sample number of digits for this example
        n_digits = rng.integers(min_digits, max_digits + 1)

        # Generate random operands
        max_val = 10 ** n_digits
        a = rng.integers(0, max_val)
        b = rng.integers(0, max_val)

        # Create input/target strings
        input_str, target_str = preprocess(a, b, max_digits=10)

        # Tokenize
        input_tokens = tokenize(input_str)
        target_tokens = tokenize(target_str) + [EOS_TOKEN]

        # Full sequence for autoregressive training
        # Input: [input_str, target_str] (without final token)
        # Target: [input_str, target_str, EOS] shifted by 1
        full_seq = input_tokens + tokenize(target_str)
        full_target = input_tokens[1:] + target_tokens

        seq_length = len(full_seq)
        inputs[i, :seq_length] = full_seq
        targets[i, :len(full_target)] = full_target

        # Mask: only compute loss on target portion (after '=')
        # Position of '=' is at len(input_tokens) - 1
        eq_pos = len(input_tokens) - 1
        mask[i, eq_pos:eq_pos + len(target_tokens)] = 1.0

    return inputs, targets, mask


def generate_fixed_dataset(
    seed: int,
    size: int,
    min_digits: int = 10,
    max_digits: int = 10
) -> List[Tuple[int, int, int]]:
    """
    Generate a fixed dataset of (a, b, a+b) tuples.

    Args:
        seed: Random seed for reproducibility
        size: Number of examples
        min_digits: Minimum digits (10 for test set)
        max_digits: Maximum digits

    Returns:
        List of (a, b, c) tuples where c = a + b
    """
    rng = np.random.default_rng(seed)
    dataset = []

    for _ in range(size):
        n_digits = rng.integers(min_digits, max_digits + 1)
        max_val = 10 ** n_digits
        a = int(rng.integers(0, max_val))
        b = int(rng.integers(0, max_val))
        dataset.append((a, b, a + b))

    return dataset


def prepare_batch_from_dataset(
    dataset: List[Tuple[int, int, int]],
    indices: np.ndarray,
    seq_len: int = 35
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare a batch from a fixed dataset.

    Args:
        dataset: List of (a, b, c) tuples
        indices: Indices to select from dataset
        seq_len: Sequence length for padding

    Returns:
        (inputs, targets, mask) arrays
    """
    batch_size = len(indices)
    inputs = np.full((batch_size, seq_len), PAD_TOKEN, dtype=np.int32)
    targets = np.full((batch_size, seq_len), PAD_TOKEN, dtype=np.int32)
    mask = np.zeros((batch_size, seq_len), dtype=np.float32)

    for i, idx in enumerate(indices):
        a, b, _ = dataset[idx]

        input_str, target_str = preprocess(a, b, max_digits=10)

        input_tokens = tokenize(input_str)
        target_tokens = tokenize(target_str) + [EOS_TOKEN]

        full_seq = input_tokens + tokenize(target_str)
        full_target = input_tokens[1:] + target_tokens

        seq_length = len(full_seq)
        inputs[i, :seq_length] = full_seq
        targets[i, :len(full_target)] = full_target

        eq_pos = len(input_tokens) - 1
        mask[i, eq_pos:eq_pos + len(target_tokens)] = 1.0

    return inputs, targets, mask


if __name__ == "__main__":
    # Test the data pipeline
    print("Testing data pipeline...")

    # Test preprocess/postprocess
    a, b = 1234567890, 9876543210
    input_str, target_str = preprocess(a, b)
    print(f"Input: {input_str}")
    print(f"Target: {target_str}")

    # Verify postprocess
    result = postprocess(target_str)
    expected = a + b
    print(f"Postprocess result: {result}")
    print(f"Expected: {expected}")
    print(f"Match: {result == expected}")

    # Test batch generation
    rng = np.random.default_rng(42)
    inputs, targets, mask = generate_batch(rng, batch_size=4, min_digits=1, max_digits=3)
    print(f"\nBatch shapes: inputs={inputs.shape}, targets={targets.shape}, mask={mask.shape}")

    # Show first example
    print(f"\nFirst example:")
    print(f"  Input tokens:  {inputs[0][:25]}")
    print(f"  Target tokens: {targets[0][:25]}")
    print(f"  Mask:          {mask[0][:25]}")
    print(f"  Input string:  {detokenize(inputs[0])}")
