#!/usr/bin/env python3
"""
Worker-based orchestrator for addition experiment sweep on spot TPU VMs.

Architecture:
- Workers are VMs that are validated (spinup + install + test)
- Tasks are assigned to idle workers from a queue
- Workers complete tasks and receive new ones (no redeploy needed)
- Failed installations = hard error, delete VM and retry
- Preempted/failed tasks return to pending queue

Commands:
    init     - Initialize task matrix
    status   - Show status of workers and tasks
    run      - Main orchestration loop
    results  - Show completed results
    cleanup  - Delete all VMs

Usage:
    python -m infra.spot.addition_orchestrator init
    python -m infra.spot.addition_orchestrator run
"""

import argparse
import concurrent.futures
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from . import state as state_ops
from . import vm_ops
from .config import (
    ADDITION_SWEEP,
    GCS_RUNS_DIR,
    HEARTBEAT_TIMEOUT,
    MAX_CONCURRENT_VMS,
    POLL_INTERVAL,
    REMOTE_DIR,
)
from .orchestrator import deploy_code

# Worker states
WORKER_CREATING = "creating"
WORKER_INSTALLING = "installing"
WORKER_READY = "ready"      # Validated and idle, can accept tasks
WORKER_BUSY = "busy"        # Running a task
WORKER_FAILED = "failed"    # Hard failure, needs deletion
WORKER_PREEMPTED = "preempted"

# In-memory worker registry (rebuilt on restart via discover_existing_workers)
workers = {}  # vm_name -> {"state": ..., "task_id": ..., "created_at": ...}
workers_lock = threading.Lock()


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def discover_existing_workers(redeploy: bool = True) -> int:
    """
    Discover existing VMs and rebuild worker registry from GCS task states.
    Called at orchestrator startup to recover from restarts.

    Returns number of workers discovered.
    """
    log("[discover] Scanning for existing workers...")

    # Get all add-worker VMs
    vms = {v["name"]: v["state"] for v in vm_ops.list_vms(prefix="add-worker-")}
    if not vms:
        log("[discover] No existing workers found")
        return 0

    # Get all task states to find running tasks
    task_states = state_ops.list_states()
    running_tasks = {s["vm_name"]: s for s in task_states if s["status"] == "running" and s.get("vm_name")}

    discovered = 0
    for vm_name, vm_state in vms.items():
        if vm_state != "READY":
            log(f"  [{vm_name}] VM not ready ({vm_state}), skipping")
            continue

        # Check if this VM has a running task
        task_state = running_tasks.get(vm_name)

        if task_state:
            # VM has a running task - mark as BUSY
            task_id = task_state["task_id"]
            log(f"  [{vm_name}] Found running task: {task_id}")
            with workers_lock:
                workers[vm_name] = {
                    "state": WORKER_BUSY,
                    "task_id": task_id,
                    "created_at": datetime.now(timezone.utc),  # Unknown, use now
                }
            discovered += 1
        else:
            # VM is idle - validate and optionally redeploy code
            log(f"  [{vm_name}] No running task, validating...")

            if redeploy:
                log(f"  [{vm_name}] Redeploying latest code...")
                if not deploy_code(vm_name):
                    log(f"  [{vm_name}] Redeploy failed, marking as failed")
                    with workers_lock:
                        workers[vm_name] = {"state": WORKER_FAILED, "task_id": None, "created_at": datetime.now(timezone.utc)}
                    continue

            # Validate worker
            if validate_worker(vm_name):
                log(f"  [{vm_name}] Validated, marking as READY")
                with workers_lock:
                    workers[vm_name] = {
                        "state": WORKER_READY,
                        "task_id": None,
                        "created_at": datetime.now(timezone.utc),
                    }
                discovered += 1
            else:
                log(f"  [{vm_name}] Validation failed, marking as failed")
                with workers_lock:
                    workers[vm_name] = {"state": WORKER_FAILED, "task_id": None, "created_at": datetime.now(timezone.utc)}

    log(f"[discover] Discovered {discovered} workers ({len([w for w in workers.values() if w['state'] == WORKER_READY])} ready, {len([w for w in workers.values() if w['state'] == WORKER_BUSY])} busy)")
    return discovered


def make_vm_name(worker_id: int) -> str:
    """Generate worker VM name."""
    return f"add-worker-{worker_id}"


def find_free_worker_id() -> int:
    """Find lowest available worker ID."""
    with workers_lock:
        used = {int(name.split("-")[-1]) for name in workers.keys() if name.startswith("add-worker-")}
    for i in range(MAX_CONCURRENT_VMS * 2):  # Some buffer
        if i not in used:
            return i
    return len(used)


def validate_worker(vm_name: str) -> bool:
    """Validate worker by running a small JAX matmul test."""
    log(f"  [{vm_name}] Validating with matmul test...")
    test_cmd = (
        "python3 -c \""
        "import jax; import jax.numpy as jnp; "
        "x = jnp.ones((100, 100)); "
        "y = jnp.dot(x, x); "
        "print(f'OK: {len(jax.devices())} devices, matmul sum={float(y.sum())}')"
        "\""
    )
    rc, stdout, stderr = vm_ops.ssh_command(vm_name, test_cmd)
    if rc != 0:
        log(f"  [{vm_name}] Validation failed: {stderr}")
        return False
    log(f"  [{vm_name}] {stdout.strip()}")
    return True


def install_on_worker(vm_name: str) -> bool:
    """Install dependencies on worker. Returns False on hard failure."""
    log(f"  [{vm_name}] Installing dependencies...")
    install_cmd = (
        "pip3 install --quiet --upgrade pip && "
        "pip3 install --quiet 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && "
        "pip3 install --quiet flax optax numpy"
    )
    rc, stdout, stderr = vm_ops.ssh_command(vm_name, install_cmd)
    if rc != 0:
        log(f"  [{vm_name}] Install FAILED (hard error): {stderr}")
        return False
    log(f"  [{vm_name}] Install complete")
    return True


def spinup_worker(worker_id: int) -> Optional[str]:
    """
    Spin up a single worker: create VM, deploy code, install deps, validate.
    Returns vm_name on success, None on failure.
    """
    vm_name = make_vm_name(worker_id)
    log(f"[spinup] Starting worker {vm_name}")

    with workers_lock:
        workers[vm_name] = {"state": WORKER_CREATING, "task_id": None, "created_at": datetime.now(timezone.utc)}

    # Create VM with retry
    for attempt in range(3):
        log(f"  [{vm_name}] Creating VM (attempt {attempt + 1}/3)...")
        if vm_ops.create_vm(vm_name):
            break
        time.sleep(10)
    else:
        log(f"  [{vm_name}] VM creation failed after 3 attempts")
        with workers_lock:
            workers[vm_name]["state"] = WORKER_FAILED
        return None

    # Wait for READY
    log(f"  [{vm_name}] Waiting for VM to be ready...")
    for _ in range(30):  # 5 min max
        if vm_ops.vm_exists(vm_name):
            break
        time.sleep(10)
    else:
        log(f"  [{vm_name}] VM not ready after 5 min")
        with workers_lock:
            workers[vm_name]["state"] = WORKER_FAILED
        return None

    # Deploy code
    with workers_lock:
        workers[vm_name]["state"] = WORKER_INSTALLING

    if not deploy_code(vm_name):
        log(f"  [{vm_name}] Deploy failed (hard error)")
        with workers_lock:
            workers[vm_name]["state"] = WORKER_FAILED
        return None

    # Install dependencies
    if not install_on_worker(vm_name):
        with workers_lock:
            workers[vm_name]["state"] = WORKER_FAILED
        return None

    # Validate
    if not validate_worker(vm_name):
        with workers_lock:
            workers[vm_name]["state"] = WORKER_FAILED
        return None

    with workers_lock:
        workers[vm_name]["state"] = WORKER_READY

    log(f"[spinup] Worker {vm_name} READY")
    return vm_name


def spinup_workers_concurrent(count: int) -> list:
    """Spin up multiple workers concurrently."""
    if count <= 0:
        return []

    log(f"[spinup] Starting {count} workers concurrently...")
    worker_ids = [find_free_worker_id() + i for i in range(count)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        futures = {executor.submit(spinup_worker, wid): wid for wid in worker_ids}
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    log(f"[spinup] {len(results)}/{count} workers ready")
    return results


def assign_task_to_worker(vm_name: str, task_id: str) -> bool:
    """Assign a task to a ready worker."""
    log(f"[assign] {task_id} -> {vm_name}")

    # Update task state
    state = state_ops.get_state(task_id)
    if not state:
        log(f"  [error] Task {task_id} not found")
        return False

    state["status"] = "running"
    state["vm_name"] = vm_name
    state["started_at"] = datetime.now(timezone.utc).isoformat()
    state["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    state_ops.put_state(task_id, state)

    # Update worker state
    with workers_lock:
        workers[vm_name]["state"] = WORKER_BUSY
        workers[vm_name]["task_id"] = task_id

    # Start task runner on worker (fire and forget - don't wait for SSH)
    gcs_output = f"{GCS_RUNS_DIR}/{task_id}"
    runner_cmd = (
        f"cd {REMOTE_DIR} && python3 -u -m infra.spot.addition_task_runner "
        f"--task-id {task_id} --output {gcs_output} --skip-install "
        f">/tmp/task_runner.log 2>&1"
    )
    if not vm_ops.ssh_command_fire_and_forget(vm_name, runner_cmd):
        log(f"  [error] Failed to start task")
        state["status"] = "pending"  # Return to queue
        state["error"] = "Failed to start task runner"
        state_ops.put_state(task_id, state)
        with workers_lock:
            workers[vm_name]["state"] = WORKER_READY
            workers[vm_name]["task_id"] = None
        return False

    log(f"  [ok] Task started")
    return True


def check_workers_and_tasks():
    """Check worker health and task completion."""
    now = datetime.now(timezone.utc)
    vms = {v["name"]: v["state"] for v in vm_ops.list_vms(prefix="add-")}

    with workers_lock:
        worker_list = list(workers.items())

    for vm_name, worker in worker_list:
        # Check if VM still exists
        vm_state = vms.get(vm_name)

        if worker["state"] == WORKER_BUSY:
            task_id = worker["task_id"]
            if not task_id:
                continue

            task_state = state_ops.get_state(task_id)
            if not task_state:
                continue

            # Check if task completed
            if task_state["status"] == "completed":
                log(f"[complete] {task_id} on {vm_name}")
                with workers_lock:
                    workers[vm_name]["state"] = WORKER_READY
                    workers[vm_name]["task_id"] = None
                continue

            # Check for preemption (VM gone + stale heartbeat)
            vm_alive = vm_state == "READY"
            heartbeat_fresh = False
            if task_state.get("last_heartbeat"):
                age = (now - datetime.fromisoformat(task_state["last_heartbeat"])).total_seconds()
                heartbeat_fresh = age < HEARTBEAT_TIMEOUT

            if not vm_alive and not heartbeat_fresh:
                log(f"[preempt] {task_id} on {vm_name}")
                task_state["status"] = "pending"  # Return to queue
                task_state["preemption_count"] = task_state.get("preemption_count", 0) + 1
                state_ops.put_state(task_id, task_state)
                with workers_lock:
                    workers[vm_name]["state"] = WORKER_PREEMPTED

            # Check if task failed
            if task_state["status"] == "failed":
                log(f"[failed] {task_id} on {vm_name} - returning to queue")
                task_state["status"] = "pending"  # Return to queue for retry
                task_state["error"] = None
                state_ops.put_state(task_id, task_state)
                with workers_lock:
                    workers[vm_name]["state"] = WORKER_READY
                    workers[vm_name]["task_id"] = None

        elif worker["state"] in (WORKER_FAILED, WORKER_PREEMPTED):
            # Clean up failed/preempted workers
            if vm_name in vms:
                log(f"[cleanup] Deleting {vm_name}")
                vm_ops.delete_vm(vm_name)
            with workers_lock:
                del workers[vm_name]


def get_ready_workers() -> list:
    with workers_lock:
        return [name for name, w in workers.items() if w["state"] == WORKER_READY]


def get_pending_tasks() -> list:
    return [s for s in state_ops.list_states() if s["status"] == "pending"]


def get_worker_count() -> int:
    with workers_lock:
        return len([w for w in workers.values() if w["state"] not in (WORKER_FAILED, WORKER_PREEMPTED)])


# ============================================================================
# Commands
# ============================================================================

def cmd_init(args):
    """Initialize task matrix for addition sweep."""
    log("Initializing addition sweep tasks...")

    existing = {s["task_id"] for s in state_ops.list_states()}
    created = 0

    for task_id, n_layers, n_heads, d_model, d_ff in ADDITION_SWEEP:
        if task_id in existing:
            print(f"  [skip] {task_id}")
            continue

        state = {
            "task_id": task_id,
            "vm_name": None,
            "config": {"n_layers": n_layers, "n_heads": n_heads, "d_model": d_model, "d_ff": d_ff},
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "last_heartbeat": None,
            "step": 0,
            "best_dev_score": None,
            "preemption_count": 0,
            "error": None,
            "checkpoint": None,
            "task": "addition",
        }
        state_ops.put_state(task_id, state)
        print(f"  [create] {task_id}")
        created += 1

    print(f"\nCreated {created} tasks")


def cmd_status(args):
    """Show status of workers and tasks."""
    states = state_ops.list_states()
    vms = {v["name"]: v["state"] for v in vm_ops.list_vms(prefix="add-")}

    # Build VM -> task mapping from GCS state (works without running orchestrator)
    vm_to_task = {s["vm_name"]: s["task_id"] for s in states if s["status"] == "running" and s.get("vm_name")}

    # Task status
    by_status = {}
    for s in states:
        by_status.setdefault(s["status"], []).append(s)

    print("=== Tasks ===")
    print(f"{'Status':<12} {'Count':>5}")
    print("-" * 20)
    for status in ["completed", "running", "pending", "failed"]:
        if status in by_status:
            print(f"{status:<12} {len(by_status[status]):>5}")
    print(f"Total: {len(states)}")

    print("\n=== Workers ===")
    print(f"Active VMs: {len(vms)}")
    for name, state in sorted(vms.items()):
        # First check GCS state, then in-memory registry
        task = vm_to_task.get(name)
        if not task:
            with workers_lock:
                if name in workers:
                    w = workers[name]
                    task = w.get("task_id") or f"({w['state']})"
        task = task or "(idle)"
        print(f"  {name}: {state} -> {task}")

    if args.verbose:
        print("\n=== Task Details ===")
        for s in sorted(states, key=lambda x: x["task_id"]):
            cfg = s.get("config", {})
            cfg_str = f"{cfg.get('n_layers', '?')}L {cfg.get('d_model', '?')}d"
            score = s.get("best_dev_score")
            score_str = f"{score:.3f}" if score else "-"
            print(f"  {s['task_id']:<20} {s['status']:<12} {cfg_str:<12} {score_str}")


def cmd_run(args):
    """Main orchestration loop."""
    log("Starting worker-based orchestration (Ctrl+C to stop)...")

    # Discover existing workers on startup (handles restarts)
    discover_existing_workers(redeploy=not args.no_redeploy)

    try:
        while True:
            # Check worker health and task completion
            check_workers_and_tasks()

            # Get current state
            pending_tasks = get_pending_tasks()
            ready_workers = get_ready_workers()
            worker_count = get_worker_count()

            # FIRST: Assign tasks to any ready workers (don't wait for spinup)
            for vm_name in ready_workers:
                if not pending_tasks:
                    break
                task = pending_tasks.pop(0)
                assign_task_to_worker(vm_name, task["task_id"])

            # Refresh counts after assignment
            pending_tasks = get_pending_tasks()
            ready_workers = get_ready_workers()
            worker_count = get_worker_count()

            # THEN: Spin up more workers if needed (this blocks but work is already assigned)
            workers_needed = min(
                len(pending_tasks),  # Don't spin up more than pending tasks
                MAX_CONCURRENT_VMS - worker_count  # Respect limit
            )
            if workers_needed > 0:
                spinup_workers_concurrent(workers_needed)
                # Assign tasks to newly ready workers
                pending_tasks = get_pending_tasks()
                for vm_name in get_ready_workers():
                    if not pending_tasks:
                        break
                    task = pending_tasks.pop(0)
                    assign_task_to_worker(vm_name, task["task_id"])

            # Status summary
            states = state_ops.list_states()
            running = sum(1 for s in states if s["status"] == "running")
            completed = sum(1 for s in states if s["status"] == "completed")
            pending = sum(1 for s in states if s["status"] == "pending")

            log(f"Workers: {get_worker_count()}, Running: {running}, Pending: {pending}, Completed: {completed}/{len(states)}")

            # Check if all done
            if completed == len(states):
                log("All tasks completed!")
                break

            if pending == 0 and running == 0:
                log("No pending or running tasks. Done!")
                break

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        log("Stopped by user")


def cmd_results(args):
    """Show results from completed tasks."""
    completed = [s for s in state_ops.list_states() if s["status"] == "completed"]
    if not completed:
        print("No completed tasks")
        return

    # Sort by score descending
    completed.sort(key=lambda x: -(x.get("best_dev_score") or 0))

    print(f"\n{'Task':<20} {'Config':<15} {'Score':>8}")
    print("-" * 45)
    for s in completed:
        cfg = s.get("config", {})
        cfg_str = f"{cfg.get('n_layers')}L {cfg.get('n_heads')}H {cfg.get('d_model')}d"
        score = s.get("best_dev_score")
        score_str = f"{score:.4f}" if score else "-"
        print(f"{s['task_id']:<20} {cfg_str:<15} {score_str:>8}")


def cmd_cleanup(args):
    """Delete all worker VMs."""
    vms = [v for v in vm_ops.list_vms(prefix="add-")]
    if not vms:
        print("No VMs to clean up")
        return

    if not args.force:
        print(f"Delete {len(vms)} VMs?")
        for v in vms:
            print(f"  - {v['name']} ({v['state']})")
        if input("[y/N] ").lower() != "y":
            print("Aborted")
            return

    for v in vms:
        print(f"Deleting {v['name']}...")
        vm_ops.delete_vm(v["name"])

    print(f"Deleted {len(vms)} VMs")


def main():
    parser = argparse.ArgumentParser(description="Worker-based addition sweep orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Initialize task matrix")
    p = sub.add_parser("status", help="Show status")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("run", help="Run orchestration loop")
    p.add_argument("--no-redeploy", action="store_true", help="Skip code redeploy on existing workers")
    sub.add_parser("results", help="Show results")
    p = sub.add_parser("cleanup", help="Delete all VMs")
    p.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args()
    {"init": cmd_init, "status": cmd_status, "run": cmd_run,
     "results": cmd_results, "cleanup": cmd_cleanup}[args.cmd](args)


if __name__ == "__main__":
    main()
