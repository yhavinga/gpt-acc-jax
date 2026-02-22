"""Configuration for the addition transformer experiment."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    vocab_size: int = 14  # 0-9, +, =, <PAD>, <EOS>
    max_seq_len: int = 35  # 10 + 1 + 10 + 1 + 11 + 2
    dropout_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 256
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    # Curriculum phases: (min_digits, max_digits, steps)
    curriculum_phases: Tuple[Tuple[int, int, int], ...] = (
        (1, 3, 5000),    # Phase 1: 1-3 digits
        (1, 6, 10000),   # Phase 2: 1-6 digits
        (1, 10, 30000),  # Phase 3: 1-10 digits (full range)
    )

    # Logging
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 2000

    # Dataset sizes
    val_size: int = 5000
    test_size: int = 10000


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    model: ModelConfig
    training: TrainingConfig
    seed: int = 42
    experiment_name: str = "addition_v1"

    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()


# Token definitions
TOKENS = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '+': 10, '=': 11, '<PAD>': 12, '<EOS>': 13
}
TOKENS_INV = {v: k for k, v in TOKENS.items()}
PAD_TOKEN = TOKENS['<PAD>']
EOS_TOKEN = TOKENS['<EOS>']
