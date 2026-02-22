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
# Each entry: (task_id, n_layers, n_heads, d_model, d_ff)
# ============================================================================

ADDITION_SWEEP = [
    # Tiny models (~25-50K params)
    ("tiny-2L-32d", 2, 2, 32, 128),
    ("tiny-2L-48d", 2, 2, 48, 192),

    # Small models (~50-100K params)
    ("small-2L-64d", 2, 2, 64, 256),
    ("small-2L-64d-4h", 2, 4, 64, 256),
    ("small-3L-48d", 3, 2, 48, 192),

    # Medium-small models (~100-200K params)
    ("med-2L-96d", 2, 2, 96, 384),
    ("med-3L-64d", 3, 2, 64, 256),
    ("med-3L-64d-4h", 3, 4, 64, 256),

    # Medium models (~200-400K params)
    ("med-2L-128d", 2, 4, 128, 512),
    ("med-4L-64d", 4, 2, 64, 256),
    ("med-4L-64d-4h", 4, 4, 64, 256),

    # Larger models (~400-800K params)
    ("large-3L-128d", 3, 4, 128, 512),
    ("large-4L-128d", 4, 4, 128, 512),  # Reference: ~800K, known to work
]

# Legacy fields for backward compatibility with DUMB eval
CHECKPOINTS = []
TASKS_PHASE1 = []