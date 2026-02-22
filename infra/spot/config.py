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
# ADDITION EXPERIMENT SWEEP
# Each entry: (task_id, n_layers, n_heads, d_model, d_ff, lr, warmup_ratio)
# lr and warmup_ratio are optional (defaults: 1e-3, 0.05)
# ============================================================================

ADDITION_SWEEP = [
    # -------------------------------------------------------------------------
    # PICO SWEEP: Finding absolute minimum (sub-1K params?)
    # -------------------------------------------------------------------------

    # Ultra-tiny: can we go below 1,360?
    ("pico-1L-4d", 1, 1, 4, 16, 1e-2, 0.05),           # 468 params
    ("pico-1L-5d", 1, 1, 5, 20, 1e-2, 0.05),           # 645 params
    ("pico-1L-6d", 1, 1, 6, 24, 1e-2, 0.05),           # 846 params
    ("pico-1L-7d", 1, 1, 7, 28, 1e-2, 0.05),           # 1,071 params

    # Boundary testing around 1,360 (known smallest passing)
    ("nano-1L-8d-lr02", 1, 1, 8, 32, 2e-2, 0.05),      # 1,360 params, even higher LR
    ("nano-1L-8d-lr005", 1, 1, 8, 32, 5e-3, 0.05),     # 1,360 params, medium LR

    # Does 2-layer work with high LR?
    ("nano-2L-8d-hiLR", 2, 1, 8, 32, 1e-2, 0.05),      # 2,200 params

    # Try even higher LR for pico models
    ("pico-1L-6d-lr03", 1, 1, 6, 24, 3e-2, 0.05),      # 846 params, very high LR

    # -------------------------------------------------------------------------
    # MICRO SWEEP (completed): Finding minimum viable model size
    # -------------------------------------------------------------------------

    # Nano models (~1-3K params)
    ("nano-1L-8d", 1, 1, 8, 32, 3e-3, 0.05),           # 1,360 params - FAIL 8.4%
    ("nano-1L-8d-hiLR", 1, 1, 8, 32, 1e-2, 0.05),      # 1,360 params - PASS 100% ★
    ("nano-1L-12d", 1, 1, 12, 48, 3e-3, 0.05),         # 2,616 params - PASS 100%
    ("nano-2L-8d", 2, 1, 8, 32, 3e-3, 0.05),           # 2,200 params - FAIL 0.1%

    # Micro models (~4-8K params)
    ("micro-1L-16d", 1, 1, 16, 64, 3e-3, 0.05),        # 4,256 params - PASS 100%
    ("micro-1L-16d-hiLR", 1, 1, 16, 64, 1e-2, 0.05),   # 4,256 params - PASS 100%
    ("micro-2L-12d", 2, 1, 12, 48, 3e-3, 0.05),        # 4,452 params - FAIL 11.1%
    ("micro-2L-16d", 2, 1, 16, 64, 3e-3, 0.05),        # 7,472 params - PASS 100%
    ("micro-1L-24d", 1, 2, 24, 96, 3e-3, 0.05),        # 8,688 params - PASS 100%

    # Mini models (~10-20K params)
    ("mini-1L-32d", 1, 2, 32, 128, 1e-3, 0.05),        # 14,656 params - PASS 100%
    ("mini-2L-24d", 2, 2, 24, 96, 1e-3, 0.05),         # 15,816 params - PASS 100%
    ("mini-2L-24d-hiLR", 2, 2, 24, 96, 3e-3, 0.05),    # 15,816 params - PASS 100%

    # -------------------------------------------------------------------------
    # Reference models (known to work at 100%)
    # -------------------------------------------------------------------------
    ("tiny-2L-32d", 2, 2, 32, 128, 1e-3, 0.05),        # 27,232 params - PASS 100%
    ("small-3L-48d", 3, 2, 48, 192, 1e-3, 0.05),       # 87,360 params - PASS 100%
    ("small-2L-64d", 2, 2, 64, 256, 1e-3, 0.05),       # 103,616 params - PASS 100%
]

# Legacy fields for backward compatibility with DUMB eval
CHECKPOINTS = []
TASKS_PHASE1 = []