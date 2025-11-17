"""Generate performance reports."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger("trading_system")


class ReportGenerator:
    """Generate performance reports for trading system."""

    @staticmethod
    def generate_text_report(
        backtest_results: Dict,
        model_metrics: Dict,
        save_path: str
    ):
        """
        Generate text report.

        Args:
            backtest_results: Backtest results
            model_metrics: Model metrics
            save_path: Path to save report
        """
        logger.info("Generating text report...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ML TRADING SYSTEM - PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Model Performance
        report_lines.append("\n" + "=" * 80)
        report_lines.append("MODEL PERFORMANCE")
        report_lines.append("=" * 80)
        report_lines.append(f"Accuracy:       {model_metrics.get('accuracy', 0):.4f}")
        report_lines.append(f"Precision:      {model_metrics.get('precision', 0):.4f}")
        report_lines.append(f"Recall:         {model_metrics.get('recall', 0):.4f}")
        report_lines.append(f"F1 Score:       {model_metrics.get('f1', 0):.4f}")

        # Backtest Performance
        report_lines.append("\n" + "=" * 80)
        report_lines.append("BACKTEST PERFORMANCE")
        report_lines.append("=" * 80)
        report_lines.append(f"Initial Capital:      ${backtest_results['initial_capital']:,.2f}")
        report_lines.append(f"Final Equity:         ${backtest_results['final_equity']:,.2f}")
        report_lines.append(f"Total Return:         {backtest_results['total_return']*100:,.2f}%")
        report_lines.append(f"CAGR:                 {backtest_results['cagr']*100:,.2f}%")

        # Risk Metrics
        report_lines.append("\n" + "=" * 80)
        report_lines.append("RISK METRICS")
        report_lines.append("=" * 80)
        report_lines.append(f"Sharpe Ratio:         {backtest_results['sharpe_ratio']:.2f}")
        report_lines.append(f"Sortino Ratio:        {backtest_results['sortino_ratio']:.2f}")
        report_lines.append(f"Max Drawdown:         {backtest_results['max_drawdown']*100:.2f}%")

        # Trade Statistics
        report_lines.append("\n" + "=" * 80)
        report_lines.append("TRADE STATISTICS")
        report_lines.append("=" * 80)
        report_lines.append(f"Total Trades:         {backtest_results['total_trades']}")
        report_lines.append(f"Winning Trades:       {backtest_results['winning_trades']}")
        report_lines.append(f"Losing Trades:        {backtest_results['losing_trades']}")
        report_lines.append(f"Win Rate:             {backtest_results['win_rate']*100:.2f}%")
        report_lines.append(f"Average Win:          ${backtest_results['avg_win']:,.2f}")
        report_lines.append(f"Average Loss:         ${backtest_results['avg_loss']:,.2f}")

        pf = backtest_results['profit_factor']
        if pf == np.inf:
            report_lines.append(f"Profit Factor:        ∞ (no losing trades)")
        else:
            report_lines.append(f"Profit Factor:        {pf:.2f}")

        # Model Details
        if 'classification_report' in model_metrics:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("CLASSIFICATION REPORT")
            report_lines.append("=" * 80)
            report_lines.append(model_metrics['classification_report'])

        report_lines.append("\n" + "=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        # Save report
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Text report saved to {save_path}")

    @staticmethod
    def generate_json_report(
        backtest_results: Dict,
        model_metrics: Dict,
        config: Dict,
        save_path: str
    ):
        """
        Generate JSON report.

        Args:
            backtest_results: Backtest results
            model_metrics: Model metrics
            config: Configuration used
            save_path: Path to save report
        """
        logger.info("Generating JSON report...")

        # Remove non-serializable objects
        clean_backtest = {k: v for k, v in backtest_results.items()
                         if k not in ['trades', 'equity_curve', 'equity_dates']}

        # Convert dates to strings
        if 'equity_dates' in backtest_results:
            clean_backtest['date_range'] = {
                'start': str(backtest_results['equity_dates'][0]),
                'end': str(backtest_results['equity_dates'][-1])
            }

        clean_model = {k: v for k, v in model_metrics.items()
                      if k not in ['classification_report']}

        report = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'model_metrics': clean_model,
            'backtest_results': clean_backtest
        }

        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"JSON report saved to {save_path}")

    @staticmethod
    def generate_markdown_report(
        backtest_results: Dict,
        model_metrics: Dict,
        config: Dict,
        save_path: str
    ):
        """
        Generate Markdown report.

        Args:
            backtest_results: Backtest results
            model_metrics: Model metrics
            config: Configuration used
            save_path: Path to save report
        """
        logger.info("Generating Markdown report...")

        md_lines = []
        md_lines.append("# ML Trading System - Performance Report\n")
        md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Configuration
        md_lines.append("## Configuration\n")
        md_lines.append(f"- **Symbol:** {config.get('data', {}).get('symbol', 'N/A')}")
        md_lines.append(f"- **Timeframe:** {config.get('data', {}).get('timeframes', 'N/A')}")
        md_lines.append(f"- **Initial Capital:** ${backtest_results['initial_capital']:,.2f}")
        md_lines.append(f"- **Position Size:** {config.get('backtest', {}).get('position_size', 0)*100:.1f}%\n")

        # Model Performance
        md_lines.append("## Model Performance\n")
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")
        md_lines.append(f"| Accuracy | {model_metrics.get('accuracy', 0):.4f} |")
        md_lines.append(f"| Precision | {model_metrics.get('precision', 0):.4f} |")
        md_lines.append(f"| Recall | {model_metrics.get('recall', 0):.4f} |")
        md_lines.append(f"| F1 Score | {model_metrics.get('f1', 0):.4f} |\n")

        # Backtest Performance
        md_lines.append("## Backtest Performance\n")
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")
        md_lines.append(f"| Initial Capital | ${backtest_results['initial_capital']:,.2f} |")
        md_lines.append(f"| Final Equity | ${backtest_results['final_equity']:,.2f} |")
        md_lines.append(f"| Total Return | {backtest_results['total_return']*100:,.2f}% |")
        md_lines.append(f"| CAGR | {backtest_results['cagr']*100:,.2f}% |")
        md_lines.append(f"| Sharpe Ratio | {backtest_results['sharpe_ratio']:.2f} |")
        md_lines.append(f"| Sortino Ratio | {backtest_results['sortino_ratio']:.2f} |")
        md_lines.append(f"| Max Drawdown | {backtest_results['max_drawdown']*100:.2f}% |\n")

        # Trade Statistics
        md_lines.append("## Trade Statistics\n")
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")
        md_lines.append(f"| Total Trades | {backtest_results['total_trades']} |")
        md_lines.append(f"| Winning Trades | {backtest_results['winning_trades']} |")
        md_lines.append(f"| Losing Trades | {backtest_results['losing_trades']} |")
        md_lines.append(f"| Win Rate | {backtest_results['win_rate']*100:.2f}% |")
        md_lines.append(f"| Average Win | ${backtest_results['avg_win']:,.2f} |")
        md_lines.append(f"| Average Loss | ${backtest_results['avg_loss']:,.2f} |")

        pf = backtest_results['profit_factor']
        if pf == np.inf:
            md_lines.append(f"| Profit Factor | ∞ |")
        else:
            md_lines.append(f"| Profit Factor | {pf:.2f} |")

        md_lines.append("\n## Visualizations\n")
        md_lines.append("- Equity Curve: `equity_curve.png`")
        md_lines.append("- Drawdown: `drawdown.png`")
        md_lines.append("- Returns Distribution: `returns_distribution.png`")
        md_lines.append("- Feature Importance: `feature_importance.png`")
        md_lines.append("- Confusion Matrix: `confusion_matrix.png`")

        # Save report
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write('\n'.join(md_lines))

        logger.info(f"Markdown report saved to {save_path}")

    @staticmethod
    def generate_full_report(
        backtest_results: Dict,
        model_metrics: Dict,
        config: Dict,
        output_dir: str
    ):
        """
        Generate all report formats.

        Args:
            backtest_results: Backtest results
            model_metrics: Model metrics
            config: Configuration used
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ReportGenerator.generate_text_report(
            backtest_results,
            model_metrics,
            str(output_path / 'report.txt')
        )

        ReportGenerator.generate_json_report(
            backtest_results,
            model_metrics,
            config,
            str(output_path / 'report.json')
        )

        ReportGenerator.generate_markdown_report(
            backtest_results,
            model_metrics,
            config,
            str(output_path / 'report.md')
        )

        logger.info(f"Full report generated in {output_dir}")


# Fix import issue
import numpy as np
