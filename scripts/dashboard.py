#!/usr/bin/env python3
"""Interactive performance dashboard."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.models.experiment_tracker import ExperimentTracker


def create_dashboard(tracker: ExperimentTracker):
    """Create interactive performance dashboard."""

    df = tracker.get_experiments_df()

    if df.empty:
        print("No experiments found")
        return

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('ML Trading System - Performance Dashboard',
                 fontsize=16, fontweight='bold')

    # 1. Sharpe Ratio over time
    ax1 = fig.add_subplot(gs[0, :2])
    df.plot(x='timestamp', y='sharpe_ratio', ax=ax1, marker='o', linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Sharpe Ratio Over Time', fontweight='bold')
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Sharpe Ratio', 'Breakeven'])

    # 2. Best experiments
    ax2 = fig.add_subplot(gs[0, 2])
    top_experiments = df.nlargest(5, 'sharpe_ratio')
    ax2.barh(range(len(top_experiments)), top_experiments['sharpe_ratio'])
    ax2.set_yticks(range(len(top_experiments)))
    ax2.set_yticklabels([f"#{i+1}" for i in range(len(top_experiments))])
    ax2.set_title('Top 5 Sharpe Ratios', fontweight='bold')
    ax2.set_xlabel('Sharpe Ratio')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Returns distribution
    ax3 = fig.add_subplot(gs[1, 0])
    returns = df['total_return'] * 100
    ax3.hist(returns, bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(returns.mean(), color='r', linestyle='--',
                linewidth=2, label=f'Mean: {returns.mean():.1f}%')
    ax3.set_title('Returns Distribution', fontweight='bold')
    ax3.set_xlabel('Total Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Sharpe vs Drawdown
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(df['max_drawdown']*100, df['sharpe_ratio'],
                         c=df['total_return']*100, cmap='RdYlGn', s=100, alpha=0.6)
    ax4.set_title('Sharpe vs Max Drawdown', fontweight='bold')
    ax4.set_xlabel('Max Drawdown (%)')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Return (%)')

    # 5. Win Rate distribution
    ax5 = fig.add_subplot(gs[1, 2])
    win_rates = df['win_rate'] * 100
    ax5.boxplot(win_rates, vert=True)
    ax5.set_title('Win Rate Distribution', fontweight='bold')
    ax5.set_ylabel('Win Rate (%)')
    ax5.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)

    # 6. Model comparison
    ax6 = fig.add_subplot(gs[2, :2])
    model_perf = df.groupby('model_name').agg({
        'sharpe_ratio': 'mean',
        'total_return': 'mean',
        'max_drawdown': 'mean'
    })

    x = range(len(model_perf))
    width = 0.25

    ax6.bar([i-width for i in x], model_perf['sharpe_ratio'],
            width, label='Sharpe Ratio', alpha=0.8)
    ax6.bar(x, model_perf['total_return']*100,
            width, label='Return (%)', alpha=0.8)
    ax6.bar([i+width for i in x], -model_perf['max_drawdown']*100,
            width, label='Max DD (%)', alpha=0.8)

    ax6.set_xticks(x)
    ax6.set_xticklabels(model_perf.index, rotation=45, ha='right')
    ax6.set_title('Average Performance by Model', fontweight='bold')
    ax6.set_ylabel('Value')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 7. Statistics table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    stats_text = [
        f"Total Experiments: {len(df)}",
        f"",
        f"Sharpe Ratio:",
        f"  Mean: {df['sharpe_ratio'].mean():.2f}",
        f"  Best: {df['sharpe_ratio'].max():.2f}",
        f"",
        f"Total Return:",
        f"  Mean: {df['total_return'].mean()*100:.1f}%",
        f"  Best: {df['total_return'].max()*100:.1f}%",
        f"",
        f"Max Drawdown:",
        f"  Mean: {df['max_drawdown'].mean()*100:.1f}%",
        f"  Best: {df['max_drawdown'].min()*100:.1f}%",
    ]

    ax7.text(0.1, 0.9, '\n'.join(stats_text),
             transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Performance dashboard')
    parser.add_argument('--save', type=str, help='Save dashboard to file')

    args = parser.parse_args()

    tracker = ExperimentTracker()

    if len(tracker.experiments) == 0:
        print("No experiments found. Run some training first!")
        sys.exit(1)

    print("Creating dashboard...")
    fig = create_dashboard(tracker)

    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
