"""GCS state file operations for spot TPU orchestration.

Uses gcloud CLI directly - no SDK dependency.
"""

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .config import GCS_STATE_DIR


def _run_gcloud(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a gcloud command and return the result."""
    result = subprocess.run(
        ["gcloud", "storage"] + args,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"gcloud command failed: {result.stderr}")
    return result


def get_state(task_id: str) -> dict | None:
    """Get state for a task from GCS. Returns None if not found."""
    gcs_path = f"{GCS_STATE_DIR}/{task_id}.json"
    result = _run_gcloud(["cat", gcs_path], check=False)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def put_state(task_id: str, state: dict) -> None:
    """Write state for a task to GCS."""
    gcs_path = f"{GCS_STATE_DIR}/{task_id}.json"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(state, f, indent=2)
        f.flush()
        temp_path = f.name
    try:
        result = _run_gcloud(["cp", temp_path, gcs_path], check=False)
        if result.returncode != 0:
            print(f"[put_state] GCS write failed for {task_id}: {result.stderr}", flush=True)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def list_states() -> list[dict]:
    """List all state files from GCS and return their contents."""
    result = _run_gcloud(["ls", f"{GCS_STATE_DIR}/*.json"], check=False)
    if result.returncode != 0:
        return []

    states = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        # Extract task_id from path
        task_id = line.split("/")[-1].replace(".json", "")
        state = get_state(task_id)
        if state:
            states.append(state)
    return states


def delete_state(task_id: str) -> bool:
    """Delete a state file from GCS. Returns True if successful."""
    gcs_path = f"{GCS_STATE_DIR}/{task_id}.json"
    result = _run_gcloud(["rm", gcs_path], check=False)
    return result.returncode == 0


def heartbeat(task_id: str, step: int, best_score: float | None = None) -> None:
    """Update heartbeat timestamp for a running task."""
    state = get_state(task_id)
    if state is None:
        raise ValueError(f"No state found for task {task_id}")

    state["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    state["step"] = step
    if best_score is not None:
        state["best_dev_score"] = best_score
    put_state(task_id, state)


def create_initial_state(
    task_id: str,
    checkpoint: str,
    task: str,
    vm_name: str,
) -> dict:
    """Create an initial state dict for a new task."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "task_id": task_id,
        "checkpoint": checkpoint,
        "task": task,
        "vm_name": vm_name,
        "status": "pending",
        "created_at": now,
        "started_at": None,
        "last_heartbeat": None,
        "step": 0,
        "total_steps": None,
        "best_dev_score": None,
        "preemption_count": 0,
        "error": None,
    }
