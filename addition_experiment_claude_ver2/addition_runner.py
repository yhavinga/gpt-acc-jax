#!/usr/bin/env python3
"""
Task runner for addition experiment on spot TPU VMs.
Lightweight wrapper that installs deps and runs the experiment.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"  [error] {result.stderr}")
    return result


def install_dependencies():
    """Install JAX and dependencies on TPU VM."""
    print("[setup] Installing dependencies...")
    commands = [
        "pip install --upgrade pip",
        "pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
        "pip install flax optax numpy",
    ]
    for cmd in commands:
        if run_cmd(cmd).returncode != 0:
            return False
    return True


def upload_to_gcs(local_path: str, gcs_path: str) -> bool:
    """Upload to GCS."""
    result = run_cmd(f"gcloud storage cp -r {local_path} {gcs_path}")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', required=True, help='Unique task identifier')
    parser.add_argument('--output', required=True, help='GCS output path')
    parser.add_argument('--skip-install', action='store_true')
    args = parser.parse_args()

    print(f"[addition_runner] Task: {args.task_id}")
    print(f"[addition_runner] Output: {args.output}")

    # Install dependencies
    if not args.skip_install:
        if not install_dependencies():
            print("[error] Failed to install dependencies")
            sys.exit(1)

    # Verify JAX sees TPU
    try:
        import jax
        print(f"[addition_runner] JAX devices: {jax.devices()}")
    except Exception as e:
        print(f"[error] JAX import failed: {e}")
        sys.exit(1)

    # Run the experiment
    script_dir = Path(__file__).parent
    output_dir = f"/tmp/addition_output_{args.task_id}"

    cmd = f"python {script_dir}/run_addition.py --output {output_dir}"
    print(f"[addition_runner] Running: {cmd}")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("[error] Training failed")
        sys.exit(1)

    # Upload results to GCS
    print(f"[addition_runner] Uploading results to {args.output}")
    if not upload_to_gcs(f"{output_dir}/*", args.output):
        print("[error] Failed to upload results")
        sys.exit(1)

    print("[addition_runner] Done!")


if __name__ == "__main__":
    main()