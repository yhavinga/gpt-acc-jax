"""gcloud wrapper functions for TPU VM operations."""

import json
import subprocess

from .config import PROJECT, ZONE, TPU_TYPE, RUNTIME_VERSION


def _run_gcloud(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a gcloud command and return the result."""
    result = subprocess.run(
        ["gcloud"] + args,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"gcloud command failed: {result.stderr}")
    return result


def create_vm(vm_name: str) -> bool:
    """Create a spot TPU VM. Returns True if successful."""
    result = _run_gcloud(
        [
            "compute", "tpus", "tpu-vm", "create", vm_name,
            f"--zone={ZONE}",
            f"--accelerator-type={TPU_TYPE}",
            f"--version={RUNTIME_VERSION}",
            "--spot",
            f"--project={PROJECT}",
        ],
        check=False,
    )
    return result.returncode == 0


def delete_vm(vm_name: str) -> bool:
    """Delete a TPU VM. Returns True if successful."""
    result = _run_gcloud(
        [
            "compute", "tpus", "tpu-vm", "delete", vm_name,
            f"--zone={ZONE}",
            f"--project={PROJECT}",
            "--quiet",
        ],
        check=False,
    )
    return result.returncode == 0


def list_vms(prefix: str = "dumb-") -> list[dict]:
    """List TPU VMs with the given prefix. Returns list of {name, state}."""
    result = _run_gcloud(
        [
            "compute", "tpus", "tpu-vm", "list",
            f"--zone={ZONE}",
            f"--project={PROJECT}",
            "--format=json",
        ],
        check=False,
    )
    if result.returncode != 0:
        return []

    vms = json.loads(result.stdout) if result.stdout.strip() else []
    filtered = []
    for vm in vms:
        # name is full path like "projects/.../nodes/dumb-xxx", extract last part
        full_name = vm.get("name", "")
        name = full_name.split("/")[-1] if "/" in full_name else full_name
        if name.startswith(prefix):
            filtered.append({
                "name": name,
                "state": vm.get("state", "UNKNOWN"),
            })
    return filtered


def vm_exists(vm_name: str) -> bool:
    """Check if a VM exists and is in READY state."""
    result = _run_gcloud(
        [
            "compute", "tpus", "tpu-vm", "describe", vm_name,
            f"--zone={ZONE}",
            f"--project={PROJECT}",
            "--format=json",
        ],
        check=False,
    )
    if result.returncode != 0:
        return False
    vm = json.loads(result.stdout)
    return vm.get("state") == "READY"


def vm_state(vm_name: str) -> str | None:
    """Get the state of a VM. Returns None if VM doesn't exist."""
    result = _run_gcloud(
        [
            "compute", "tpus", "tpu-vm", "describe", vm_name,
            f"--zone={ZONE}",
            f"--project={PROJECT}",
            "--format=json",
        ],
        check=False,
    )
    if result.returncode != 0:
        return None
    vm = json.loads(result.stdout)
    return vm.get("state")


def ssh_command(vm_name: str, command: str, timeout: int | None = None) -> tuple[int, str, str]:
    """Run a command on a VM via SSH. Returns (returncode, stdout, stderr)."""
    args = [
        "compute", "tpus", "tpu-vm", "ssh", vm_name,
        f"--zone={ZONE}",
        f"--project={PROJECT}",
        "--command", command,
    ]
    try:
        result = subprocess.run(
            ["gcloud"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 0, "", ""  # Timeout is OK for background commands


def ssh_command_fire_and_forget(vm_name: str, command: str) -> bool:
    """
    Start a background command on VM and return immediately.
    Uses timeout to avoid waiting for SSH to close file descriptors.
    Returns True if command was sent (doesn't guarantee execution).
    """
    # Wrap in bash -c and use setsid to detach from controlling terminal
    # The bash -c ensures the entire command runs in the new session
    escaped_cmd = command.replace("'", "'\\''")
    bg_command = f"setsid bash -c '{escaped_cmd}' </dev/null >/dev/null 2>&1 &"
    # 5 second timeout - enough to start the process
    rc, _, _ = ssh_command(vm_name, bg_command, timeout=5)
    return rc == 0


def scp_to_vm(vm_name: str, local_path: str, remote_path: str) -> bool:
    """Copy a file or directory to a VM. Returns True if successful."""
    result = _run_gcloud(
        [
            "compute", "tpus", "tpu-vm", "scp",
            "--recurse",
            local_path,
            f"{vm_name}:{remote_path}",
            f"--zone={ZONE}",
            f"--project={PROJECT}",
        ],
        check=False,
    )
    return result.returncode == 0


def scp_from_vm(vm_name: str, remote_path: str, local_path: str) -> bool:
    """Copy a file or directory from a VM. Returns True if successful."""
    result = _run_gcloud(
        [
            "compute", "tpus", "tpu-vm", "scp",
            "--recurse",
            f"{vm_name}:{remote_path}",
            local_path,
            f"--zone={ZONE}",
            f"--project={PROJECT}",
        ],
        check=False,
    )
    return result.returncode == 0
