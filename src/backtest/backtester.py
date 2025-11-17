"""Backtesting engine with transaction costs and risk management."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("trading_system")


@dataclass
class Trade:
    """Trade record."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    size: float
    direction: str  # 'long' or 'short'
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    status: str = 'open'  # 'open' or 'closed'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class Backtester:
    """Backtest trading strategies."""

    def __init__(self, config: dict):
        """
        Initialize backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission = config.get('costs', {}).get('commission', 0.001)
        self.slippage = config.get('costs', {}).get('slippage', 0.0005)

        # Risk management
        self.position_size = config.get('position_size', 0.02)
        self.stop_loss_pct = config.get('risk_management', {}).get('stop_loss', 0.02)
        self.take_profit_pct = config.get('risk_management', {}).get('take_profit', 0.04)
        self.max_drawdown = config.get('risk_management', {}).get('max_drawdown', 0.20)
        self.max_positions = config.get('risk_management', {}).get('max_positions', 1)

        # State
        self.reset()

    def reset(self):
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.equity = [self.initial_capital]
        self.equity_dates = []
        self.trades: List[Trade] = []
        self.open_positions: List[Trade] = []
        self.peak_equity = self.initial_capital
        self.max_dd_encountered = 0.0

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        prices: Optional[pd.Series] = None
    ) -> Dict:
        """
        Run backtest.

        Args:
            df: DataFrame with OHLCV data
            signals: Trading signals (0=sell/short, 1=hold, 2=buy/long)
            prices: Execution prices (if None, uses 'close')

        Returns:
            Dictionary with backtest results
        """
        logger.info("Running backtest...")
        self.reset()

        if prices is None:
            prices = df['close']

        # Ensure signals and prices are aligned
        common_idx = signals.index.intersection(prices.index)
        signals = signals.loc[common_idx]
        prices = prices.loc[common_idx]

        for i, (timestamp, signal) in enumerate(signals.items()):
            price = prices.loc[timestamp]

            # Update equity and check drawdown
            current_equity = self._calculate_equity(price)
            self.equity.append(current_equity)
            self.equity_dates.append(timestamp)

            # Check peak and drawdown
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity

            current_dd = (self.peak_equity - current_equity) / self.peak_equity
            self.max_dd_encountered = max(self.max_dd_encountered, current_dd)

            # Emergency stop if max drawdown exceeded
            if current_dd > self.max_drawdown:
                logger.warning(f"Max drawdown exceeded at {timestamp}. Closing all positions.")
                self._close_all_positions(price, timestamp)
                continue

            # Check stop loss and take profit for open positions
            self._check_risk_management(price, timestamp)

            # Execute new trades based on signals
            if signal == 2:  # Buy signal
                if len(self.open_positions) < self.max_positions:
                    self._open_position(price, timestamp, 'long')

            elif signal == 0:  # Sell signal
                # Close long positions
                for pos in self.open_positions[:]:
                    if pos.direction == 'long':
                        self._close_position(pos, price, timestamp)

        # Close all remaining positions at the end
        final_price = prices.iloc[-1]
        final_time = prices.index[-1]
        self._close_all_positions(final_price, final_time)

        # Calculate metrics
        results = self._calculate_metrics()

        logger.info(f"Backtest complete. Final equity: ${results['final_equity']:.2f}")
        logger.info(f"Total return: {results['total_return']*100:.2f}%")
        logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")

        return results

    def _open_position(self, price: float, timestamp: datetime, direction: str):
        """Open a new position."""
        # Calculate position size
        position_value = self.capital * self.position_size
        execution_price = price * (1 + self.slippage if direction == 'long' else 1 - self.slippage)

        # Account for commission
        commission_cost = position_value * self.commission
        size = (position_value - commission_cost) / execution_price

        # Calculate stop loss and take profit
        if direction == 'long':
            stop_loss = execution_price * (1 - self.stop_loss_pct)
            take_profit = execution_price * (1 + self.take_profit_pct)
        else:
            stop_loss = execution_price * (1 + self.stop_loss_pct)
            take_profit = execution_price * (1 - self.take_profit_pct)

        trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=execution_price,
            exit_price=None,
            size=size,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.open_positions.append(trade)
        self.capital -= (size * execution_price + commission_cost)

        logger.debug(f"Opened {direction} position at {execution_price:.2f} (size: {size:.4f})")

    def _close_position(self, trade: Trade, price: float, timestamp: datetime):
        """Close an existing position."""
        execution_price = price * (1 - self.slippage if trade.direction == 'long' else 1 + self.slippage)

        # Calculate PnL
        if trade.direction == 'long':
            pnl = (execution_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - execution_price) * trade.size

        # Subtract commission
        commission_cost = execution_price * trade.size * self.commission
        pnl -= commission_cost

        # Update capital
        self.capital += (execution_price * trade.size)

        # Update trade
        trade.exit_time = timestamp
        trade.exit_price = execution_price
        trade.pnl = pnl
        trade.return_pct = pnl / (trade.entry_price * trade.size)
        trade.status = 'closed'

        self.trades.append(trade)
        self.open_positions.remove(trade)

        logger.debug(f"Closed {trade.direction} position at {execution_price:.2f} (PnL: ${pnl:.2f})")

    def _close_all_positions(self, price: float, timestamp: datetime):
        """Close all open positions."""
        for pos in self.open_positions[:]:
            self._close_position(pos, price, timestamp)

    def _check_risk_management(self, price: float, timestamp: datetime):
        """Check stop loss and take profit for open positions."""
        for pos in self.open_positions[:]:
            if pos.direction == 'long':
                if price <= pos.stop_loss:
                    logger.debug(f"Stop loss triggered at {price:.2f}")
                    self._close_position(pos, price, timestamp)
                elif price >= pos.take_profit:
                    logger.debug(f"Take profit triggered at {price:.2f}")
                    self._close_position(pos, price, timestamp)

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open positions."""
        equity = self.capital

        for pos in self.open_positions:
            if pos.direction == 'long':
                unrealized_pnl = (current_price - pos.entry_price) * pos.size
            else:
                unrealized_pnl = (pos.entry_price - current_price) * pos.size

            equity += unrealized_pnl

        return equity

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        closed_trades = [t for t in self.trades if t.status == 'closed']

        if not closed_trades:
            logger.warning("No closed trades. Metrics may be incomplete.")

        # Basic metrics
        final_equity = self.equity[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Trade metrics
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # Risk metrics
        returns = pd.Series(self.equity).pct_change().dropna()
        sharpe_ratio = self._calculate_sharpe(returns)
        sortino_ratio = self._calculate_sortino(returns)
        max_drawdown = self._calculate_max_drawdown()

        # Additional metrics
        profit_factor = (
            abs(sum([t.pnl for t in winning_trades])) /
            abs(sum([t.pnl for t in losing_trades]))
            if losing_trades and abs(sum([t.pnl for t in losing_trades])) > 0 else np.inf
        )

        # CAGR
        days = (self.equity_dates[-1] - self.equity_dates[0]).days
        years = days / 365.25
        cagr = (final_equity / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0

        metrics = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': self.equity,
            'equity_dates': self.equity_dates,
            'trades': closed_trades
        }

        return metrics

    @staticmethod
    def _calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()

        return sharpe

    @staticmethod
    def _calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()

        return sortino

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        equity_series = pd.Series(self.equity)
        cumulative_max = equity_series.expanding().max()
        drawdown = (equity_series - cumulative_max) / cumulative_max

        return abs(drawdown.min())

    def get_trade_log(self) -> pd.DataFrame:
        """
        Get trade log as DataFrame.

        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'direction': trade.direction,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct,
                'duration': (trade.exit_time - trade.entry_time).total_seconds() / 3600 if trade.exit_time else None
            })

        return pd.DataFrame(trades_data)
