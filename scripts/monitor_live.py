#!/usr/bin/env python3
"""Monitor live trading performance in real-time."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import json
from datetime import datetime
from src.utils.logger import setup_logger


def load_trading_log(log_file: str):
    """Load and parse trading log."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        return lines
    except FileNotFoundError:
        return []


def parse_status_line(line: str):
    """Parse status line from trading log."""
    if "Status -" in line:
        parts = line.split("Status -")[1].strip()
        status = {}
        for item in parts.split(","):
            if ":" in item:
                key, value = item.split(":", 1)
                status[key.strip()] = value.strip()
        return status
    return None


def display_dashboard(log_file: str):
    """Display live trading dashboard."""
    print("\033[2J\033[H")  # Clear screen

    print("="*80)
    print("LIVE TRADING MONITOR")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log: {log_file}")
    print("="*80)

    lines = load_trading_log(log_file)

    if not lines:
        print("\nNo log data available")
        return

    # Get recent status lines
    recent_statuses = []
    for line in reversed(lines):
        status = parse_status_line(line)
        if status:
            recent_statuses.append(status)
        if len(recent_statuses) >= 10:
            break

    if recent_statuses:
        latest = recent_statuses[0]

        print("\nCURRENT STATUS:")
        print("-"*80)
        for key, value in latest.items():
            print(f"{key:15s}: {value}")

        print("\nRECENT HISTORY (Last 10 updates):")
        print("-"*80)
        print(f"{'Time':20s} {'Price':12s} {'Signal':8s} {'Position':10s} {'Daily PnL':12s}")
        print("-"*80)

        for i, status in enumerate(recent_statuses):
            print(f"{status.get('Time', 'N/A'):20s} "
                  f"{status.get('Price', 'N/A'):12s} "
                  f"{status.get('Signal', 'N/A'):8s} "
                  f"{status.get('Position', 'N/A'):10s} "
                  f"{status.get('Daily PnL', 'N/A'):12s}")

    # Check for errors or warnings
    errors = [l for l in lines[-50:] if 'ERROR' in l or 'WARNING' in l]
    if errors:
        print("\nRECENT WARNINGS/ERRORS:")
        print("-"*80)
        for error in errors[-5:]:
            print(error.strip())

    print("\n" + "="*80)
    print("Press CTRL+C to exit")


def main():
    parser = argparse.ArgumentParser(description='Monitor live trading')
    parser.add_argument('--log-file', type=str, default='data/logs/live_trading.log')
    parser.add_argument('--refresh', type=int, default=5, help='Refresh interval (seconds)')

    args = parser.parse_args()

    try:
        while True:
            display_dashboard(args.log_file)
            time.sleep(args.refresh)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")


if __name__ == '__main__':
    main()
