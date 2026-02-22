#!/bin/bash
# Launch addition experiment on a spot TPU VM
# Usage: ./launch_spot.sh [experiment_name]

set -e

EXPERIMENT_NAME="${1:-addition-v1}"
PROJECT="YOUR_GCP_PROJECT"
ZONE="us-central2-b"
TPU_TYPE="v4-8"
VM_NAME="addition-${EXPERIMENT_NAME}"
GCS_OUTPUT="gs://YOUR_GCS_BUCKET/addition-experiments/${EXPERIMENT_NAME}"

echo "=== Addition Experiment Launcher ==="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "VM Name: ${VM_NAME}"
echo "Output: ${GCS_OUTPUT}"
echo ""

# Create tarball of the experiment code
echo "[1/5] Creating code tarball..."
TARBALL="/tmp/addition_code.tar.gz"
tar -czf ${TARBALL} \
    -C "$(dirname "$0")" \
    run_addition.py addition_runner.py

# Upload to GCS
echo "[2/5] Uploading code to GCS..."
GCS_CODE="gs://YOUR_GCS_BUCKET/addition-experiments/code/${EXPERIMENT_NAME}.tar.gz"
gcloud storage cp ${TARBALL} ${GCS_CODE}

# Create spot TPU VM
echo "[3/5] Creating spot TPU VM ${VM_NAME}..."
gcloud compute tpus tpu-vm create ${VM_NAME} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --accelerator-type=${TPU_TYPE} \
    --version=tpu-ubuntu2204-base \
    --spot \
    || { echo "VM creation failed"; exit 1; }

# Wait for VM to be ready
echo "[4/5] Waiting for VM to be ready..."
for i in {1..30}; do
    STATE=$(gcloud compute tpus tpu-vm describe ${VM_NAME} \
        --project=${PROJECT} --zone=${ZONE} \
        --format="value(state)" 2>/dev/null || echo "UNKNOWN")
    if [ "$STATE" = "READY" ]; then
        echo "VM is ready!"
        break
    fi
    echo "  State: ${STATE}, waiting..."
    sleep 10
done

# Deploy and run
echo "[5/5] Deploying and starting experiment..."
gcloud compute tpus tpu-vm ssh ${VM_NAME} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --command="
        set -e
        mkdir -p ~/addition_experiment
        cd ~/addition_experiment
        gcloud storage cp ${GCS_CODE} /tmp/code.tar.gz
        tar -xzf /tmp/code.tar.gz

        # Run in background with nohup
        nohup python addition_runner.py \
            --task-id ${EXPERIMENT_NAME} \
            --output ${GCS_OUTPUT} \
            > /tmp/addition.log 2>&1 &

        echo 'Experiment started! Logs at /tmp/addition.log'
    "

echo ""
echo "=== Experiment Launched! ==="
echo "Monitor with:"
echo "  gcloud compute tpus tpu-vm ssh ${VM_NAME} --project=${PROJECT} --zone=${ZONE} --command='tail -f /tmp/addition.log'"
echo ""
echo "Results will be at: ${GCS_OUTPUT}"
echo ""
echo "To delete VM when done:"
echo "  gcloud compute tpus tpu-vm delete ${VM_NAME} --project=${PROJECT} --zone=${ZONE} --quiet"