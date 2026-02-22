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
    # MICRO SWEEP: Finding minimum viable model size
    # -------------------------------------------------------------------------

    # Nano models (~1-3K params) - likely need higher LR
    ("nano-1L-8d", 1, 1, 8, 32, 3e-3, 0.05),           # 1,320 params
    ("nano-1L-8d-hiLR", 1, 1, 8, 32, 1e-2, 0.05),      # same with higher LR
    ("nano-1L-12d", 1, 1, 12, 48, 3e-3, 0.05),         # 2,556 params
    ("nano-2L-8d", 2, 1, 8, 32, 3e-3, 0.05),           # 2,120 params

    # Micro models (~4-8K params)
    ("micro-1L-16d", 1, 1, 16, 64, 3e-3, 0.05),        # 4,176 params
    ("micro-1L-16d-hiLR", 1, 1, 16, 64, 1e-2, 0.05),   # same with higher LR
    ("micro-2L-12d", 2, 1, 12, 48, 3e-3, 0.05),        # 4,332 params
    ("micro-2L-16d", 2, 1, 16, 64, 3e-3, 0.05),        # 7,312 params
    ("micro-1L-24d", 1, 2, 24, 96, 3e-3, 0.05),        # 8,568 params

    # Mini models (~10-20K params)
    ("mini-1L-32d", 1, 2, 32, 128, 1e-3, 0.05),        # 14,496 params
    ("mini-2L-24d", 2, 2, 24, 96, 1e-3, 0.05),         # 15,576 params
    ("mini-2L-24d-hiLR", 2, 2, 24, 96, 3e-3, 0.05),    # same with higher LR

    # -------------------------------------------------------------------------
    # Reference models (known to work at 100%)
    # -------------------------------------------------------------------------
    ("tiny-2L-32d", 2, 2, 32, 128, 1e-3, 0.05),        # 27K - proven 100%
    ("small-3L-48d", 3, 2, 48, 192, 1e-3, 0.05),       # 87K - proven 100%
    ("small-2L-64d", 2, 2, 64, 256, 1e-3, 0.05),       # 104K - proven 100%
]

# Legacy fields for backward compatibility with DUMB eval
CHECKPOINTS = []
TASKS_PHASE1 = []