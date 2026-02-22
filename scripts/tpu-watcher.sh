#!/bin/bash
# Monitor all queued TPU resources and notify on state changes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
else
    echo "Error: $ENV_FILE not found"
    echo "Create it with:"
    echo "  BOT_TOKEN=your-telegram-bot-token"
    echo "  CHAT_ID=your-chat-id"
    exit 1
fi

ZONE="${1:-us-central2-b}"
PROJECT="${2:-YOUR_GCP_PROJECT}"

STATE_FILE="/tmp/.tpu-watcher-states"

send_telegram() {
    curl -s "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d "chat_id=${CHAT_ID}" \
        -d "text=$1" > /dev/null
}

get_states() {
    gcloud compute tpus queued-resources list \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --format='value(name,state.state)' 2>/dev/null | sort || true
}

echo "Watching TPU queued-resources in $ZONE"
echo "Will notify Telegram on state changes"
echo "Press Ctrl+C to stop"
echo ""

# Initialize state file
get_states > "$STATE_FILE"
echo "Initial states:"
cat "$STATE_FILE"
echo ""

while true; do
    sleep 60

    CURRENT=$(get_states)
    TIMESTAMP=$(date '+%H:%M:%S')

    # Compare with previous
    while IFS=$'\t' read -r name state; do
        prev_state=$(grep "^${name}"$'\t' "$STATE_FILE" 2>/dev/null | cut -f2 || echo "")

        if [[ -n "$prev_state" && "$prev_state" != "$state" ]]; then
            echo "$TIMESTAMP: $name: $prev_state → $state"

            if [[ "$state" == "ACTIVE" ]]; then
                send_telegram "🎉 TPU '$name' is now ACTIVE!"
            elif [[ "$state" == "FAILED" ]]; then
                send_telegram "❌ TPU '$name' FAILED"
            else
                send_telegram "📡 TPU '$name': $prev_state → $state"
            fi
        fi
    done <<< "$CURRENT"

    # Check for new resources
    while IFS=$'\t' read -r name state; do
        if ! grep -q "^${name}"$'\t' "$STATE_FILE" 2>/dev/null; then
            echo "$TIMESTAMP: NEW $name ($state)"
            send_telegram "🆕 New TPU '$name' ($state)"
        fi
    done <<< "$CURRENT"

    # Save current state
    echo "$CURRENT" > "$STATE_FILE"

    echo "$TIMESTAMP: checked $(echo "$CURRENT" | wc -l | tr -d ' ') resources"
done