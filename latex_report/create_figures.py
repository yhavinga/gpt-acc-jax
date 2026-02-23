#!/usr/bin/env python3
"""Create publication-quality figures for the LaTeX report."""

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

# Color palette - colorblind friendly
COLORS = {
    'winner': '#2ecc71',      # Green - new winner
    'previous': '#3498db',    # Blue - previous winner
    'baseline': '#9b59b6',    # Purple - baseline
    'partial': '#f39c12',     # Orange - partial success
    'failed': '#e74c3c',      # Red - failed
    'sinusoidal': '#95a5a6',  # Gray - sinusoidal failures
}

def load_data():
    """Load wandb data from JSON."""
    with open(os.path.join(os.path.dirname(__file__), 'wandb_data.json')) as f:
        return json.load(f)


def get_run_by_name(data, name):
    """Find a run by name."""
    for run in data:
        if run['name'] == name:
            return run
    return None


def fig1_grokking_curves(data, output_dir):
    """Figure 1: Validation accuracy curves showing the grokking moment."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Key runs to show
    runs_to_plot = [
        ('pico-7d-ff14-lr02', '777 params (winner)', COLORS['winner'], 2.0),
        ('pico-1L-7d-both', '973 params', COLORS['previous'], 1.5),
        ('nano-1L-8d-hiLR', '1,360 params', COLORS['baseline'], 1.5),
        ('pico-1L-6d-tied', '792 params (partial)', COLORS['partial'], 1.5),
        ('pico-7d-ff14', '777 params (low LR)', COLORS['failed'], 1.0),
    ]

    for run_name, label, color, linewidth in runs_to_plot:
        run = get_run_by_name(data, run_name)
        if run and run['history']['val_accuracy']:
            steps = [pt['step'] for pt in run['history']['val_accuracy']]
            accs = [pt['value'] for pt in run['history']['val_accuracy']]
            ax.plot(steps, accs, label=label, color=color, linewidth=linewidth)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Grokking: Sudden Accuracy Jumps During Training')
    ax.legend(loc='lower right')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 27500)

    # Add vertical lines for curriculum phases
    ax.axvline(x=2000, color='gray', linestyle=':', alpha=0.5, label='Phase 1→2')
    ax.axvline(x=7000, color='gray', linestyle=':', alpha=0.5, label='Phase 2→3')

    # Annotate grokking moment
    ax.annotate('Grokking\n(step 14k)', xy=(14000, 0.88), xytext=(17000, 0.6),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_grokking.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig1_grokking.png'))
    plt.close()
    print("Created fig1_grokking.pdf")


def fig2_parameter_cliff(data, output_dir):
    """Figure 2: Final accuracy vs parameter count (the cliff)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Extract data points
    params = []
    accs = []
    names = []
    colors = []

    for run in data:
        n_params = run['summary'].get('n_params')
        test_acc = run['summary'].get('final_test_accuracy')
        if n_params and test_acc is not None:
            params.append(n_params)
            accs.append(test_acc)
            names.append(run['name'])

            # Color by category
            if 'femto' in run['name'] or 'sinusoidal' in str(run['config']):
                colors.append(COLORS['sinusoidal'])
            elif test_acc >= 0.99:
                if n_params < 800:
                    colors.append(COLORS['winner'])
                else:
                    colors.append(COLORS['baseline'])
            elif test_acc >= 0.5:
                colors.append(COLORS['partial'])
            else:
                colors.append(COLORS['failed'])

    # Plot scatter
    ax.scatter(params, accs, c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Highlight key points
    for run_name, marker, size in [('pico-7d-ff14-lr02', '*', 200), ('pico-1L-7d-both', 's', 100)]:
        run = get_run_by_name(data, run_name)
        if run:
            ax.scatter([run['summary']['n_params']], [run['summary']['final_test_accuracy']],
                      marker=marker, s=size, c=COLORS['winner'] if 'ff14' in run_name else COLORS['previous'],
                      edgecolors='black', linewidth=1, zorder=10)

    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('The Parameter Cliff: Sharp Transition at ~800 Parameters')
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)

    # Add cliff region shading
    ax.axvspan(700, 1000, alpha=0.1, color='red', label='Transition zone')

    # Add threshold line
    ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(50000, 0.96, '99% threshold', fontsize=8, color='green')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['winner'], markersize=8, label='≥99% (winner)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['baseline'], markersize=8, label='≥99%'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['partial'], markersize=8, label='50-99%'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['failed'], markersize=8, label='<50%'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['sinusoidal'], markersize=8, label='Sinusoidal pos'),
    ]
    ax.legend(handles=legend_elements, loc='center right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_parameter_cliff.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig2_parameter_cliff.png'))
    plt.close()
    print("Created fig2_parameter_cliff.pdf")


def fig3_lr_comparison(data, output_dir):
    """Figure 3: Learning rate impact - same model, different LR."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: pico-7d-ff14 vs pico-7d-ff14-lr02
    ax = axes[0]
    for run_name, label, color in [
        ('pico-7d-ff14', 'LR=0.01 (failed)', COLORS['failed']),
        ('pico-7d-ff14-lr02', 'LR=0.02 (winner)', COLORS['winner']),
    ]:
        run = get_run_by_name(data, run_name)
        if run and run['history']['val_accuracy']:
            steps = [pt['step'] for pt in run['history']['val_accuracy']]
            accs = [pt['value'] for pt in run['history']['val_accuracy']]
            ax.plot(steps, accs, label=label, color=color, linewidth=1.5)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('777 Parameters: LR Makes or Breaks')
    ax.legend(loc='center right')
    ax.set_ylim(-0.05, 1.05)

    # Right: nano-1L-8d variants
    ax = axes[1]
    for run_name, label, color in [
        ('nano-1L-8d', 'LR=0.003 (failed)', COLORS['failed']),
        ('nano-1L-8d-lr005', 'LR=0.005', COLORS['partial']),
        ('nano-1L-8d-hiLR', 'LR=0.01 (success)', COLORS['winner']),
    ]:
        run = get_run_by_name(data, run_name)
        if run and run['history']['val_accuracy']:
            steps = [pt['step'] for pt in run['history']['val_accuracy']]
            accs = [pt['value'] for pt in run['history']['val_accuracy']]
            ax.plot(steps, accs, label=label, color=color, linewidth=1.5)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('1,360 Parameters: LR Threshold')
    ax.legend(loc='center right')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_lr_comparison.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig3_lr_comparison.png'))
    plt.close()
    print("Created fig3_lr_comparison.pdf")


def fig4_architecture_comparison(data, output_dir):
    """Figure 4: 1-layer vs 2-layer at same param count."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Compare 1L vs 2L
    comparisons = [
        ('nano-1L-8d-hiLR', '1L, d=8 (1,360 params)', COLORS['winner']),
        ('nano-2L-8d-hiLR', '2L, d=8 (2,200 params)', COLORS['failed']),
        ('nano-1L-12d', '1L, d=12 (2,616 params)', COLORS['baseline']),
        ('micro-2L-12d', '2L, d=12 (4,452 params)', COLORS['partial']),
    ]

    for run_name, label, color in comparisons:
        run = get_run_by_name(data, run_name)
        if run and run['history']['val_accuracy']:
            steps = [pt['step'] for pt in run['history']['val_accuracy']]
            accs = [pt['value'] for pt in run['history']['val_accuracy']]
            ax.plot(steps, accs, label=label, color=color, linewidth=1.5)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('1-Layer Models Beat 2-Layer at Same Scale')
    ax.legend(loc='center right', fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_architecture.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig4_architecture.png'))
    plt.close()
    print("Created fig4_architecture.pdf")


def fig5_optimization_failures(data, output_dir):
    """Figure 5: Why certain optimizations fail."""
    fig, ax = plt.subplots(figsize=(7, 4))

    # Group by optimization type
    categories = []
    accuracies = []
    colors_list = []

    # Baseline (winner)
    categories.append('Baseline\n(973 params)')
    run = get_run_by_name(data, 'pico-1L-7d-both')
    accuracies.append(run['summary']['final_test_accuracy'] if run else 0)
    colors_list.append(COLORS['winner'])

    # Winner
    categories.append('2x FFN\n(777 params)')
    run = get_run_by_name(data, 'pico-7d-ff14-lr02')
    accuracies.append(run['summary']['final_test_accuracy'] if run else 0)
    colors_list.append(COLORS['winner'])

    # RMSNorm
    categories.append('RMSNorm\n(952 params)')
    run = get_run_by_name(data, 'pico-7d-rms')
    accuracies.append(run['summary']['final_test_accuracy'] if run else 0)
    colors_list.append(COLORS['partial'])

    # No-delimiter
    categories.append('No delimiters\n(917 params)')
    run = get_run_by_name(data, 'pico-7d-nodlm')
    accuracies.append(run['summary']['final_test_accuracy'] if run else 0)
    colors_list.append(COLORS['failed'])

    # Sinusoidal
    categories.append('Sinusoidal pos\n(728 params)')
    run = get_run_by_name(data, 'femto-7d-sin')
    accuracies.append(run['summary']['final_test_accuracy'] if run else 0)
    colors_list.append(COLORS['sinusoidal'])

    # All optimizations (sinusoidal)
    categories.append('All failed opts\n(679 params)')
    run = get_run_by_name(data, 'femto-7d-full')
    accuracies.append(run['summary']['final_test_accuracy'] if run else 0)
    colors_list.append(COLORS['sinusoidal'])

    bars = ax.bar(categories, accuracies, color=colors_list, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Test Accuracy')
    ax.set_title('What Works and What Breaks')
    ax.set_ylim(0, 1.1)

    # Add threshold line
    ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(5.5, 1.02, '99%', fontsize=8, color='green')

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_optimizations.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig5_optimizations.png'))
    plt.close()
    print("Created fig5_optimizations.pdf")


def fig6_loss_curves(data, output_dir):
    """Figure 6: Training loss curves."""
    fig, ax = plt.subplots(figsize=(6, 4))

    runs_to_plot = [
        ('pico-7d-ff14-lr02', '777 params (winner)', COLORS['winner']),
        ('pico-1L-7d-both', '973 params', COLORS['previous']),
        ('nano-1L-8d-hiLR', '1,360 params', COLORS['baseline']),
    ]

    for run_name, label, color in runs_to_plot:
        run = get_run_by_name(data, run_name)
        if run and run['history']['train_loss']:
            steps = [pt['step'] for pt in run['history']['train_loss']]
            losses = [pt['value'] for pt in run['history']['train_loss']]
            ax.plot(steps, losses, label=label, color=color, linewidth=1.5)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves')
    ax.legend(loc='upper right')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_loss.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig6_loss.png'))
    plt.close()
    print("Created fig6_loss.pdf")


def main():
    print("Loading wandb data...")
    data = load_data()

    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating figures in {output_dir}")
    print("=" * 50)

    fig1_grokking_curves(data, output_dir)
    fig2_parameter_cliff(data, output_dir)
    fig3_lr_comparison(data, output_dir)
    fig4_architecture_comparison(data, output_dir)
    fig5_optimization_failures(data, output_dir)
    fig6_loss_curves(data, output_dir)

    print("=" * 50)
    print("All figures created successfully!")


if __name__ == "__main__":
    main()
