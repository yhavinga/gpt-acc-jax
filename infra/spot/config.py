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
# ============================================================================

ADDITION_SWEEP = [
    # -------------------------------------------------------------------------
    # ULTRA-PICO: Sub-1K params with architectural optimizations
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
    # NANO/MICRO: Learning rate experiments
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