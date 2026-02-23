#!/usr/bin/env python3
"""Fetch all experiment data from wandb for the addition-sweep project."""

import json
import os
import pandas as pd
import wandb

# Initialize wandb API
api = wandb.Api()

# Project configuration
ENTITY = None  # Will use default entity from wandb config
PROJECT = "addition-sweep"

def fetch_all_runs():
    """Fetch all runs from the wandb project."""
    runs = api.runs(f"{PROJECT}" if ENTITY is None else f"{ENTITY}/{PROJECT}")

    all_data = []

    for run in runs:
        print(f"Fetching: {run.name} ({run.state})")

        # Get run config
        config = dict(run.config)

        # Get summary metrics
        summary = dict(run.summary)

        # Get history (loss and accuracy curves)
        history_data = {
            "train_loss": [],
            "val_accuracy": []
        }

        try:
            history = run.history(keys=["train/loss", "val/accuracy"])
            for _, row in history.iterrows():
                step = row.get('_step')
                loss = row.get('train/loss')
                acc = row.get('val/accuracy')

                if step is not None and not pd.isna(loss):
                    history_data["train_loss"].append({"step": int(step), "value": float(loss)})
                if step is not None and not pd.isna(acc):
                    history_data["val_accuracy"].append({"step": int(step), "value": float(acc)})
        except Exception as e:
            print(f"  Warning: Could not fetch history for {run.name}: {e}")

        # Extract key metrics
        run_data = {
            "name": run.name,
            "state": run.state,
            "config": config,
            "summary": {
                "final_test_accuracy": summary.get("final/test_accuracy"),
                "final_val_accuracy": summary.get("final/val_accuracy"),
                "n_params": summary.get("n_params"),
            },
            "history": history_data
        }

        all_data.append(run_data)
        print(f"  -> test_acc={run_data['summary']['final_test_accuracy']}, n_params={run_data['summary']['n_params']}")

    return all_data


def main():
    print(f"Fetching runs from wandb project: {PROJECT}")
    print("=" * 60)

    data = fetch_all_runs()

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), "wandb_data.json")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print("=" * 60)
    print(f"Saved {len(data)} runs to {output_path}")

    # Print summary
    successful = [d for d in data if d['summary']['final_test_accuracy'] is not None and d['summary']['final_test_accuracy'] >= 0.99]
    print(f"\nSuccessful runs (>=99% accuracy): {len(successful)}")
    for run in sorted(successful, key=lambda x: x['summary'].get('n_params', float('inf'))):
        print(f"  {run['name']}: {run['summary']['n_params']} params, {run['summary']['final_test_accuracy']:.4f} acc")


if __name__ == "__main__":
    main()
