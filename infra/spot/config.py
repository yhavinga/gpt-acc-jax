"""Configuration constants for spot TPU VM orchestration.

Edit this file to configure your experiment sweep.
"""

# GCS paths
GCS_BUCKET = "YOUR_GCS_BUCKET"
GCS_PREFIX = "addition-sweep"
GCS_STATE_DIR = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/state"
GCS_RUNS_DIR = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/runs"
GCS_RESULTS_DIR = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/results"

# TPU VM configuration - using TRC spot allotments
# Available zones:
#   us-central2-b: 32 v4 chips (v4-8 = 1 VM, v4-32 = 4 workers)
#   us-central1-a: 64 v5e chips
#   europe-west4-b: 64 v5e chips
#   europe-west4-a: 64 v6e chips
#   us-east1-d: 64 v6e chips

PROJECT = "YOUR_GCP_PROJECT"
ZONE = "us-central2-b"
TPU_TYPE = "v4-8"
RUNTIME_VERSION = "tpu-ubuntu2204-base"

# Orchestration settings
MAX_CONCURRENT_VMS = 4
HEARTBEAT_INTERVAL = 60
HEARTBEAT_TIMEOUT = 300
POLL_INTERVAL = 60

# Remote directory on TPU VMs
REMOTE_DIR = "~/gpt-acc-jax"

# ============================================================================
# ADDITION EXPERIMENT SWEEP RESULTS
#
# WINNER: pico-1L-7d-both achieves 100% accuracy with only 973 parameters!
#
# Key findings:
# - 1 layer always beats 2 layers at same param count
# - Tied embeddings + no FFN bias together enable sub-1K params
# - Higher learning rate (0.01+) essential for tiny models
# - d=7 is minimum viable hidden dimension
#
# Format: (task_id, n_layers, n_heads, d_model, d_ff, lr, warmup_ratio)
# Extended: (task_id, n_layers, n_heads, d_model, d_ff, lr, warmup, ffn_bias, tied)
# Dict format: {"task_id": ..., "n_layers": ..., ...} for full control
#
# New optimizations available:
# - sinusoidal: Use fixed sinusoidal positional embeddings (saves d_model * seq_len params)
# - rmsnorm: Use RMSNorm instead of LayerNorm (saves 1 param per norm layer)
# - no_delim: Fixed-format input without +/=/PAD/EOS tokens (vocab 10 vs 14)
# - tied_qk: Share Q and K projections (saves d_model^2 params per layer)
# - rope: Use Rotary Position Embeddings (no learnable position params)
# ============================================================================

ADDITION_SWEEP = [
    # =========================================================================
    # PICO-V3 SWEEP: Smaller FFN with learned positions
    # =========================================================================
    #
    # The 973-param winner uses d_ff=28 (4x). Let's try smaller FFN ratios.
    # FFN params = d_model * d_ff * 2 = 7 * d_ff * 2
    #   d_ff=28: 392 params
    #   d_ff=21: 294 params (saves 98)
    #   d_ff=14: 196 params (saves 196)
    #   d_ff=7:  98 params (saves 294) - minimum, 1x ratio
    #

    # d_ff=21 (3x ratio) - Expected: ~875 params
    {"task_id": "pico-7d-ff21", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 21,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True},

    # d_ff=14 (2x ratio) - Expected: ~777 params
    {"task_id": "pico-7d-ff14", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 14,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True},

    # d_ff=7 (1x ratio, minimal FFN) - Expected: ~679 params
    {"task_id": "pico-7d-ff7", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 7,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True},

    # Try higher LR for smaller FFN
    {"task_id": "pico-7d-ff14-lr02", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 14,
     "lr": 2e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True},

    {"task_id": "pico-7d-ff7-lr02", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 7,
     "lr": 2e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True},

    # Also try d=6 with longer training (more steps in curriculum)
    # Maybe d=6 just needs more capacity elsewhere or more training
    {"task_id": "pico-6d-ff24-lr02", "n_layers": 1, "n_heads": 1, "d_model": 6, "d_ff": 24,
     "lr": 2e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True},

    # =========================================================================
    # PICO-V2 SWEEP: Optimizations with LEARNED positions (COMPLETED - FAILED)
    # =========================================================================
    # RESULTS:
    #   pico-7d-rms: 43.8% - RMSNorm breaks it
    #   pico-7d-nodlm: 9.6% - No-delimiter breaks it
    #   pico-7d-rms-nodlm: 1.1% - Both together = worse
    #   pico-7d-rope-*: FAILED - RoPE crashes
    #

    # RMSNorm only (with learned positions) - saves 21 params
    # Expected: 952 params
    {"task_id": "pico-7d-rms", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "rmsnorm": True},

    # No-delimiter format only (with learned positions) - saves 28 params
    # Expected: 945 params
    {"task_id": "pico-7d-nodlm", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "no_delim": True},

    # RMSNorm + no-delimiter (with learned positions) - saves ~49 params
    # Expected: ~924 params
    {"task_id": "pico-7d-rms-nodlm", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "rmsnorm": True, "no_delim": True},

    # RoPE (Rotary Position Embeddings) - no learnable position params
    # Expected: 728 params (same as sinusoidal but different mechanism)
    {"task_id": "pico-7d-rope", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "rope": True},

    # RoPE + RMSNorm
    # Expected: 707 params
    {"task_id": "pico-7d-rope-rms", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "rope": True, "rmsnorm": True},

    # RoPE + no-delimiter
    # Expected: 700 params
    {"task_id": "pico-7d-rope-nodlm", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "rope": True, "no_delim": True},

    # RoPE + RMSNorm + no-delimiter (full combo)
    # Expected: 679 params
    {"task_id": "pico-7d-rope-full", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "rope": True, "rmsnorm": True, "no_delim": True},

    # =========================================================================
    # FEMTO-SWEEP: Ultra-optimized models (COMPLETED - ALL FAILED)
    # =========================================================================
    # RESULT: Sinusoidal positions break the model entirely at small scales
    #
    # Optimization stack (cumulative savings from 973 baseline):
    #   sinusoidal positions: -245 params (removes learned pos embeddings)
    #   RMSNorm: -21 params (no bias in normalization)
    #   no_delim (vocab=10): -28 params (smaller token embedding)
    #   tied Q=K: -49 params (shares Q and K projection)
    #   d_ff=14 (2x): -196 params (smaller FFN)
    #

    # Baseline: Current winner for comparison
    # pico-1L-7d-both: 973 params

    # Sinusoidal positions only: removes 245 learned position params
    # Expected: 728 params
    {"task_id": "femto-7d-sin", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "sinusoidal": True},

    # Sinusoidal + RMSNorm (no bias in norms)
    # Expected: 707 params
    {"task_id": "femto-7d-sin-rms", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "sinusoidal": True, "rmsnorm": True},

    # Sinusoidal + no-delimiter format: vocab 10 vs 14
    # Expected: 700 params (approx)
    {"task_id": "femto-7d-sin-nodlm", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "sinusoidal": True, "no_delim": True},

    # All safe optimizations: sinusoidal + rmsnorm + no-delimiter
    # Expected: 679 params
    {"task_id": "femto-7d-full", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "sinusoidal": True, "rmsnorm": True, "no_delim": True},

    # Try tied Q=K (risky but could save 49 params)
    # Expected: 630 params
    {"task_id": "femto-7d-tiedqk", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 28,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "sinusoidal": True, "rmsnorm": True, "no_delim": True, "tied_qk": True},

    # Smaller FFN (2x instead of 4x): big savings
    # Expected: 483 params
    {"task_id": "femto-7d-ff2x", "n_layers": 1, "n_heads": 1, "d_model": 7, "d_ff": 14,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "sinusoidal": True, "rmsnorm": True, "no_delim": True},

    # Push to d=6 with all optimizations
    # Expected: 510 params
    {"task_id": "femto-6d-full", "n_layers": 1, "n_heads": 1, "d_model": 6, "d_ff": 24,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "sinusoidal": True, "rmsnorm": True, "no_delim": True},

    # d=6 with 2x FFN - most aggressive viable
    # Expected: 366 params
    {"task_id": "femto-6d-ff2x", "n_layers": 1, "n_heads": 1, "d_model": 6, "d_ff": 12,
     "lr": 1e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "sinusoidal": True, "rmsnorm": True, "no_delim": True},

    # Try higher LR for smaller models
    {"task_id": "femto-6d-lr02", "n_layers": 1, "n_heads": 1, "d_model": 6, "d_ff": 24,
     "lr": 2e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "sinusoidal": True, "rmsnorm": True, "no_delim": True},

    # Push to d=5 with all optimizations
    # Expected: 365 params
    {"task_id": "femto-5d-full", "n_layers": 1, "n_heads": 1, "d_model": 5, "d_ff": 20,
     "lr": 2e-2, "warmup": 0.05, "ffn_bias": False, "tied_emb": True,
     "sinusoidal": True, "rmsnorm": True, "no_delim": True},

    # -------------------------------------------------------------------------
    # ULTRA-PICO: Sub-1K params with architectural optimizations (COMPLETED)
    # -------------------------------------------------------------------------

    # d=7 variants - WINNER HERE!
    ("pico-1L-7d-both", 1, 1, 7, 28, 1e-2, 0.05, False, True),    # 973 - PASS 100% ★
    ("pico-1L-7d-nob", 1, 1, 7, 28, 1e-2, 0.05, False, False),    # 1,071 - PASS 100%
    ("pico-1L-7d-tied", 1, 1, 7, 28, 1e-2, 0.05, True, True),     # 1,008 - FAIL 0.8%
    ("pico-1L-7d", 1, 1, 7, 28, 1e-2, 0.05),                      # 1,106 - PASS 100%

    # d=6 variants - all fail, d=6 is below threshold
    ("pico-1L-6d-tied", 1, 1, 6, 24, 1e-2, 0.05, True, True),     # 792 - FAIL 72.5%
    ("pico-1L-6d-nob", 1, 1, 6, 24, 1e-2, 0.05, False, False),    # 846 - FAIL 9.8%
    ("pico-1L-6d-both", 1, 1, 6, 24, 1e-2, 0.05, False, True),    # 762 - FAIL 0.9%
    ("pico-1L-6d", 1, 1, 6, 24, 1e-2, 0.05),                      # 876 - FAIL 0.1%
    ("pico-1L-6d-lr03", 1, 1, 6, 24, 3e-2, 0.05),                 # 876 - FAIL 9.8%

    # d=5 and d=4 - too small
    ("pico-1L-5d-both", 1, 1, 5, 20, 1e-2, 0.05, False, True),    # 575 - FAIL 0%
    ("pico-1L-5d", 1, 1, 5, 20, 1e-2, 0.05),                      # 670 - FAIL 0.6%
    ("pico-1L-4d", 1, 1, 4, 16, 1e-2, 0.05),                      # 488 - FAIL 0%

    # -------------------------------------------------------------------------
    # NANO/MICRO: Learning rate experiments (COMPLETED)
    # -------------------------------------------------------------------------

    # d=8 variants - LR matters!
    ("nano-1L-8d-hiLR", 1, 1, 8, 32, 1e-2, 0.05),                 # 1,360 - PASS 100%
    ("nano-1L-8d-lr02", 1, 1, 8, 32, 2e-2, 0.05),                 # 1,360 - PASS 100%
    ("nano-1L-8d-lr005", 1, 1, 8, 32, 5e-3, 0.05),                # 1,360 - PASS 100%
    ("nano-1L-8d", 1, 1, 8, 32, 3e-3, 0.05),                      # 1,360 - FAIL 8.4%

    # 2-layer models - always fail at nano scale
    ("nano-2L-8d-hiLR", 2, 1, 8, 32, 1e-2, 0.05),                 # 2,200 - FAIL 0.1%
    ("nano-2L-8d", 2, 1, 8, 32, 3e-3, 0.05),                      # 2,200 - FAIL 0.1%

    # Larger nano/micro
    ("nano-1L-12d", 1, 1, 12, 48, 3e-3, 0.05),                    # 2,616 - PASS 100%
    ("micro-1L-16d", 1, 1, 16, 64, 3e-3, 0.05),                   # 4,256 - PASS 100%
    ("micro-1L-16d-hiLR", 1, 1, 16, 64, 1e-2, 0.05),              # 4,256 - PASS 100%
    ("micro-2L-12d", 2, 1, 12, 48, 3e-3, 0.05),                   # 4,452 - FAIL 11.1%
    ("micro-2L-16d", 2, 1, 16, 64, 3e-3, 0.05),                   # 7,472 - PASS 100%
    ("micro-1L-24d", 1, 2, 24, 96, 3e-3, 0.05),                   # 8,688 - PASS 100%

    # -------------------------------------------------------------------------
    # MINI/SMALL/MEDIUM: Reference models (all pass)
    # -------------------------------------------------------------------------
    ("mini-1L-32d", 1, 2, 32, 128, 1e-3, 0.05),                   # 14,656 - PASS 100%
    ("mini-2L-24d", 2, 2, 24, 96, 1e-3, 0.05),                    # 15,816 - PASS 100%
    ("mini-2L-24d-hiLR", 2, 2, 24, 96, 3e-3, 0.05),               # 15,816 - PASS 100%
    ("tiny-2L-32d", 2, 2, 32, 128, 1e-3, 0.05),                   # 27,232 - PASS 100%
    ("small-3L-48d", 3, 2, 48, 192, 1e-3, 0.05),                  # 87,360 - PASS 100%
    ("small-2L-64d", 2, 2, 64, 256, 1e-3, 0.05),                  # 103,616 - PASS 100%
]

# Legacy fields for backward compatibility with DUMB eval
CHECKPOINTS = []
TASKS_PHASE1 = []