#!/bin/bash
# Single TPU management script for gpt-acc-jax project
# TRC spot VM allotment
set -Eeuo pipefail

#############################################################################
# CONFIGURATION - Edit these for your project
#############################################################################

PROJECT="${PROJECT:-YOUR_GCP_PROJECT}"
ZONE="${ZONE:-us-central2-b}"
TPU_TYPE="${TPU_TYPE:-v4-8}"  # Default, can be overridden with --tpu-type or env var
TPU_NAME="${TPU_NAME:-tpu-v4-gpt-acc}"
RUNTIME_VERSION="${RUNTIME_VERSION:-tpu-ubuntu2204-base}"
SPOT_INSTANCE="${SPOT_INSTANCE:-true}"  # true for spot (TRC allotment), false for on-demand
REMOTE_DIR="${REMOTE_DIR:-~/gpt-acc-jax}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

#############################################################################
# COLORS AND LOGGING
#############################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
log_step()    { echo -e "\n${CYAN}==>${NC} $1\n"; }

#############################################################################
# HELPER FUNCTIONS
#############################################################################

is_multi_host() {
    # v4-32 and larger are multi-host (pod slices)
    # v4-8 is single host
    [[ "$TPU_TYPE" == "v4-32" || "$TPU_TYPE" == "v4-64" || "$TPU_TYPE" == "v4-128" || \
       "$TPU_TYPE" == "v5p-"* || "$TPU_TYPE" == "v5litepod-"* ]]
}

get_num_workers() {
    # Return number of workers based on TPU type
    case "$TPU_TYPE" in
        v4-8)   echo 1 ;;
        v4-32)  echo 4 ;;
        v4-64)  echo 8 ;;
        v4-128) echo 16 ;;
        *)      echo 1 ;;  # Default to single worker
    esac
}

check_gcloud_auth() {
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not authenticated with gcloud. Run: gcloud auth login"
    fi
    log_info "Authenticated as: $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
}

check_tpu_exists() {
    gcloud compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --format="value(name)" 2>/dev/null || true
}

get_tpu_state() {
    gcloud compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --format="value(state)" 2>/dev/null || echo "NOT_FOUND"
}

get_tpu_ip() {
    # Try external IP first
    local ip=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --format="value(networkEndpoints[0].accessConfig.externalIp)" 2>/dev/null || true)

    if [[ -z "$ip" || "$ip" == "None" ]]; then
        # Fall back to internal IP
        ip=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --format="value(networkEndpoints[0].ipAddress)" 2>/dev/null || true)
    fi

    echo "$ip"
}

update_ssh_config() {
    local ip="$1"
    cat > ssh_config << EOF
Host $TPU_NAME
    HostName $ip
    User $USER
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    IdentityFile ~/.ssh/google_compute_engine
EOF
    log_success "SSH config updated (IP: $ip)"
}

run_on_tpu() {
    local cmd="$1"
    if is_multi_host; then
        # Multi-host: run on all workers via gcloud
        gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --worker=all \
            --command="$cmd"
    else
        # Single host: use SSH config
        ssh -F ssh_config "$TPU_NAME" "$cmd"
    fi
}

#############################################################################
# MAIN COMMANDS
#############################################################################

cmd_status() {
    log_step "TPU Status"

    local state=$(get_tpu_state)

    if [[ "$state" == "NOT_FOUND" ]]; then
        log_warning "TPU '$TPU_NAME' not found in zone $ZONE"
        return 1
    fi

    log_info "Name: $TPU_NAME"
    log_info "Zone: $ZONE"
    log_info "Project: $PROJECT"
    log_info "Type: $TPU_TYPE"

    if is_multi_host; then
        local num_workers=$(get_num_workers)
        log_info "Mode: Multi-host ($num_workers workers)"
    else
        log_info "Mode: Single-host"
    fi

    case "$state" in
        READY)
            log_success "State: $state"
            local ip=$(get_tpu_ip)
            log_info "IP: $ip"
            ;;
        CREATING|PROVISIONING|RESTARTING)
            log_warning "State: $state (still provisioning...)"
            ;;
        PREEMPTED)
            log_error "State: $state (spot instance was terminated, run: ./tpu.sh provision)"
            ;;
        *)
            log_warning "State: $state"
            ;;
    esac
}

cmd_provision() {
    log_step "Provisioning TPU"

    local state=$(get_tpu_state)

    if [[ "$state" == "READY" ]]; then
        log_success "TPU already exists and is ready!"
        update_ssh_config "$(get_tpu_ip)"
        return 0
    elif [[ "$state" == "PREEMPTED" ]]; then
        log_warning "TPU was preempted, creating new instance..."
    elif [[ "$state" != "NOT_FOUND" ]]; then
        log_warning "TPU exists in state: $state"
        echo -n "Use existing TPU? [Y/n]: "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            return 0
        fi
    fi

    # Create queued resource
    local queued_name="${TPU_NAME}-queued-$(date +%s)"
    log_info "Creating queued resource: $queued_name"

    local cmd=(gcloud compute tpus queued-resources create "$queued_name"
        --node-id="$TPU_NAME"
        --project="$PROJECT"
        --zone="$ZONE"
        --accelerator-type="$TPU_TYPE"
        --runtime-version="$RUNTIME_VERSION")

    if [[ "$SPOT_INSTANCE" == "true" ]]; then
        cmd+=(--spot)
        log_info "Creating spot instance (can be preempted)"
    else
        log_info "Creating on-demand instance"
    fi

    "${cmd[@]}" --quiet

    # Wait for provisioning
    log_info "Waiting for TPU to be ready (this may take 5-30 minutes)..."
    local max_wait=1800  # 30 minutes
    local start_time=$(date +%s)

    while true; do
        local current_state=$(get_tpu_state)

        case "$current_state" in
            READY)
                log_success "TPU is ready!"
                break
                ;;
            CREATING|PROVISIONING)
                echo -n "."
                ;;
            FAILED)
                log_error "TPU provisioning failed!"
                ;;
            *)
                echo -n "."
                ;;
        esac

        local elapsed=$(($(date +%s) - start_time))
        if [[ $elapsed -gt $max_wait ]]; then
            log_error "Timeout waiting for TPU provisioning"
        fi

        sleep 5
    done

    echo ""
    update_ssh_config "$(get_tpu_ip)"
}

cmd_deploy() {
    log_step "Deploying Code to TPU"

    # Check TPU is ready
    local state=$(get_tpu_state)
    if [[ "$state" != "READY" ]]; then
        log_error "TPU not ready (state: $state). Run: ./tpu.sh provision"
    fi

    # Create remote directory on all workers
    run_on_tpu "mkdir -p $REMOTE_DIR"

    if is_multi_host; then
        # Multi-host: use gcloud scp to all workers
        local num_workers=$(get_num_workers)
        log_info "Syncing files to $num_workers TPU workers..."

        # Create a tarball excluding unnecessary files (faster than scp --recurse)
        log_info "Creating archive..."
        local tarball="/tmp/gpt-acc-deploy-$$.tar.gz"
        tar czf "$tarball" \
            --exclude='experiments' \
            --exclude='analysis' \
            --exclude='checkpoints' \
            --exclude='./checkpoint-*' \
            --exclude='artifacts' \
            --exclude='ssh_config' \
            --exclude='google-cloud-sdk' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.git' \
            --exclude='.idea' \
            --exclude='venv' \
            --exclude='*.tar.gz' \
            --exclude='*.pdf' \
            .

        # Copy tarball and extract on each worker
        for worker in $(seq 0 $((num_workers - 1))); do
            log_info "Deploying to worker $worker..."
            gcloud compute tpus tpu-vm scp \
                --zone="$ZONE" --project="$PROJECT" --worker="$worker" \
                "$tarball" "${TPU_NAME}:/tmp/gpt-acc-deploy.tar.gz"

            gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
                --zone="$ZONE" --project="$PROJECT" --worker="$worker" \
                --command="cd $REMOTE_DIR && tar xzf /tmp/gpt-acc-deploy.tar.gz && rm /tmp/gpt-acc-deploy.tar.gz"
        done

        rm -f "$tarball"

        # Copy auth tokens to all workers
        if [[ -f "$HOME/.cache/huggingface/token" ]]; then
            log_info "Copying HuggingFace token to all workers..."
            run_on_tpu "mkdir -p ~/.cache/huggingface"
            for worker in $(seq 0 $((num_workers - 1))); do
                gcloud compute tpus tpu-vm scp \
                    --zone="$ZONE" --project="$PROJECT" --worker="$worker" \
                    "$HOME/.cache/huggingface/token" "${TPU_NAME}:~/.cache/huggingface/"
            done
        fi

        if [[ -f "$HOME/.netrc" ]]; then
            log_info "Copying Wandb credentials to all workers..."
            for worker in $(seq 0 $((num_workers - 1))); do
                gcloud compute tpus tpu-vm scp \
                    --zone="$ZONE" --project="$PROJECT" --worker="$worker" \
                    "$HOME/.netrc" "${TPU_NAME}:~/"
            done
            run_on_tpu "chmod 600 ~/.netrc"
        fi
    else
        # Single host: use SSH config + rsync
        # Update SSH config if needed
        if [[ ! -f ssh_config ]]; then
            update_ssh_config "$(get_tpu_ip)"
        fi

        log_info "Syncing files to TPU..."
        rsync -avz --progress --delete \
            --exclude-from='.rsyncignore' \
            --exclude='experiments/' \
            --exclude='analysis/' \
            --exclude='checkpoint*/' \
            --exclude='ssh_config' \
            --exclude='google-cloud-sdk/' \
            --rsh="ssh -F ssh_config" \
            . "${TPU_NAME}:${REMOTE_DIR}/"

        # Copy auth tokens if they exist
        if [[ -f "$HOME/.cache/huggingface/token" ]]; then
            log_info "Copying HuggingFace token..."
            run_on_tpu "mkdir -p ~/.cache/huggingface"
            scp -F ssh_config "$HOME/.cache/huggingface/token" "${TPU_NAME}:~/.cache/huggingface/"
        fi

        if [[ -f "$HOME/.netrc" ]]; then
            log_info "Copying Wandb credentials..."
            scp -F ssh_config "$HOME/.netrc" "${TPU_NAME}:~/"
            run_on_tpu "chmod 600 ~/.netrc"
        fi
    fi

    log_success "Code deployed successfully!"
}

cmd_setup() {
    log_step "Setting Up TPU Environment"

    # Check if already setup
    if run_on_tpu "cd $REMOTE_DIR && [[ -d venv ]] && source venv/bin/activate && python -c 'import jax' 2>/dev/null"; then
        log_success "Environment already setup!"
        cmd_test
        return 0
    fi

    log_info "Installing Python ${PYTHON_VERSION}..."
    run_on_tpu "
        if ! command -v python${PYTHON_VERSION} &> /dev/null; then
            sudo add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
            sudo apt-get update > /dev/null 2>&1
            sudo apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-distutils > /dev/null 2>&1
        fi
    "

    log_info "Creating virtual environment..."
    run_on_tpu "cd $REMOTE_DIR && python${PYTHON_VERSION} -m venv venv"

    log_info "Installing JAX for TPU (this may take a few minutes)..."
    run_on_tpu "cd $REMOTE_DIR && source venv/bin/activate && pip install --upgrade pip setuptools wheel > /dev/null 2>&1"

    # Install JAX 0.8.0 for all TPU types
    # Force uninstall first to avoid version conflicts with pre-installed JAX
    log_info "Installing JAX 0.8.0 for TPU..."
    run_on_tpu "cd $REMOTE_DIR && source venv/bin/activate && \
        pip uninstall -y jax jaxlib libtpu libtpu-nightly 2>/dev/null || true && \
        pip install 'jax[tpu]==0.8.0' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

    log_info "Installing project requirements..."
    run_on_tpu "cd $REMOTE_DIR && source venv/bin/activate && \
        pip install -r requirements_tpu.txt > /dev/null 2>&1"

    # Install project package if setup.py or pyproject.toml exists
    if run_on_tpu "cd $REMOTE_DIR && [[ -f setup.py || -f pyproject.toml ]]"; then
        log_info "Installing project package..."
        run_on_tpu "cd $REMOTE_DIR && source venv/bin/activate && \
            pip install -e . > /dev/null 2>&1"
    fi

    log_success "Setup complete!"
    cmd_test
}

cmd_test() {
    log_step "Testing TPU Setup"

    run_on_tpu "cd $REMOTE_DIR && source venv/bin/activate && python -c '
import jax
import numpy as np

print(f\"JAX version: {jax.__version__}\")
print(f\"Devices: {jax.devices()}\")
print(f\"Device count: {jax.device_count()}\")
print(f\"Default backend: {jax.default_backend()}\")

# Simple computation test
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000, 1000))
y = jax.numpy.dot(x, x.T)
print(f\"Computation test passed! Matrix shape: {y.shape}\")
'"

    if [[ $? -eq 0 ]]; then
        log_success "All tests passed!"
    else
        log_error "Tests failed!"
    fi
}

cmd_ssh() {
    log_step "Connecting to TPU"

    local worker="${1:-0}"

    if is_multi_host; then
        local num_workers=$(get_num_workers)
        log_info "Connecting to $TPU_NAME worker $worker (of $num_workers)..."
        log_info "To activate environment: source $REMOTE_DIR/venv/bin/activate"
        gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --worker="$worker"
    else
        if [[ ! -f ssh_config ]]; then
            update_ssh_config "$(get_tpu_ip)"
        fi

        log_info "Connecting to $TPU_NAME..."
        log_info "To activate environment: source $REMOTE_DIR/venv/bin/activate"
        ssh -F ssh_config "$TPU_NAME"
    fi
}

cmd_destroy() {
    log_step "Destroying TPU Resources"

    echo -e "${RED}WARNING: This will destroy:${NC}"
    echo "  • TPU VM: $TPU_NAME"
    echo "  • All data on the TPU"
    echo "  • SSH configuration"
    echo ""
    echo -n "Are you sure? Type 'yes' to confirm: "
    read -r response

    if [[ "$response" != "yes" ]]; then
        log_warning "Cancelled"
        return 0
    fi

    # Delete TPU
    log_info "Deleting TPU VM..."
    gcloud compute tpus tpu-vm delete "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --quiet 2>/dev/null || true

    # Delete queued resources
    log_info "Deleting queued resources..."
    local queued_resources=$(gcloud compute tpus queued-resources list \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --filter="tpu.nodeSpec[0].nodeId:$TPU_NAME" \
        --format="value(name)" 2>/dev/null || true)

    if [[ -n "$queued_resources" ]]; then
        for resource in $queued_resources; do
            gcloud compute tpus queued-resources delete "$resource" \
                --zone="$ZONE" \
                --project="$PROJECT" \
                --quiet 2>/dev/null || true
        done
    fi

    # Remove SSH config
    rm -f ssh_config

    log_success "TPU resources destroyed"
}

cmd_logs() {
    log_step "TPU Logs"

    run_on_tpu "sudo journalctl -u google-startup-scripts.service -n 100"
}

cmd_run() {
    # Run a command on the TPU
    local cmd="${1:-}"
    if [[ -z "$cmd" ]]; then
        log_error "Usage: ./tpu.sh run 'command to run on TPU'"
    fi

    run_on_tpu "cd $REMOTE_DIR && source venv/bin/activate && PYTHONPATH=\$PWD/src:\$PYTHONPATH $cmd"
}

show_help() {
    local num_workers=$(get_num_workers)
    local multi_host_info=""
    if is_multi_host; then
        multi_host_info=" (multi-host: $num_workers workers)"
    fi

    echo -e "$(cat << EOF
${CYAN}TPU Management for gpt-acc-jax${NC}

${GREEN}Usage:${NC}
  ./tpu.sh [--tpu-type TYPE] [command] [options]

${GREEN}Global Options:${NC}
  ${CYAN}--tpu-type TYPE${NC}  TPU accelerator type (default: v4-8)
                    Examples: v4-8 (single), v4-32 (4 workers), v4-64 (8 workers)

${GREEN}Commands:${NC}
  ${CYAN}all${NC}        Full setup: provision + deploy + setup
  ${CYAN}status${NC}     Show TPU status
  ${CYAN}provision${NC}  Create or connect to TPU
  ${CYAN}deploy${NC}     Sync code to TPU
  ${CYAN}setup${NC}      Install dependencies on TPU
  ${CYAN}test${NC}       Test JAX and TPU setup
  ${CYAN}ssh${NC}        Connect to TPU via SSH (use: ssh [worker] for multi-host)
  ${CYAN}run${NC}        Run command on TPU (all workers for multi-host)
  ${CYAN}logs${NC}       Show TPU system logs
  ${CYAN}destroy${NC}    Delete TPU (saves money!)

${GREEN}Examples:${NC}
  ./tpu.sh status                           # Check v4-8 status
  ./tpu.sh --tpu-type v4-32 provision       # Create v4-32 pod slice
  ./tpu.sh --tpu-type v4-32 run 'python train.py'  # Run on all 4 workers

${GREEN}Current Configuration:${NC}
  • Project: $PROJECT
  • Zone: $ZONE
  • TPU: $TPU_NAME ($TPU_TYPE)${multi_host_info}
  • Spot: $SPOT_INSTANCE
EOF
)"
}

#############################################################################
# MAIN ENTRY POINT
#############################################################################

main() {
    # Parse global options first (before command)
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --tpu-type)
                TPU_TYPE="$2"
                shift 2
                ;;
            --tpu-type=*)
                TPU_TYPE="${1#*=}"
                shift
                ;;
            *)
                break  # Not a global option, must be a command
                ;;
        esac
    done

    # Check gcloud auth
    check_gcloud_auth

    # Parse command
    local cmd="${1:-}"
    shift || true

    case "$cmd" in
        "")
            # Default: show help
            show_help
            ;;
        all)
            # Full setup: provision + deploy + setup
            cmd_provision
            cmd_deploy
            cmd_setup
            ;;
        status)
            cmd_status
            ;;
        provision)
            cmd_provision
            ;;
        deploy)
            cmd_deploy
            if [[ "${1:-}" == "--setup" ]]; then
                cmd_setup
            fi
            ;;
        setup)
            cmd_setup
            ;;
        test)
            cmd_test
            ;;
        ssh)
            cmd_ssh "$@"  # Pass worker argument if provided
            ;;
        run)
            cmd_run "$@"
            ;;
        logs)
            cmd_logs
            ;;
        destroy)
            cmd_destroy
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $cmd\nRun: ./tpu.sh help"
            ;;
    esac
}

# Run main function
main "$@"