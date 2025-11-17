# ML Trading System

Ein vollständiges, selbstlernendes Machine-Learning-Trading-System für Kryptowährungen mit Backtesting, automatischer Optimierung und Live-Trading-Unterstützung.

## Features

### Core Features
- **Data Pipeline**: Automatischer Download und Bereinigung von OHLCV-Daten von Binance
- **Feature Engineering**: 50+ technische Indikatoren (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)
- **ML Models**: Support für Multiple Modelle (Logistic Regression, Random Forest, XGBoost, LightGBM)
- **Hyperparameter Tuning**: Automatische Optimierung mit Optuna oder GridSearch
- **Backtesting Engine**: Realistische Simulation mit Transaction Costs, Slippage und Risk Management
- **Evaluation**: Umfangreiche Performance-Metriken (Sharpe, Sortino, Max Drawdown, etc.)
- **Experiment Tracking**: JSON-basiertes Logging aller Experimente
- **AutoML Loop**: Automatisches Retraining und Model Selection
- **Live Trading**: Paper- und Live-Trading mit Safety Controls

### Risk Management
- Stop-Loss und Take-Profit pro Trade
- Maximum Drawdown Protection
- Daily Loss Limits
- Position Sizing (Fixed Fraction, Kelly Criterion ready)
- Emergency Stop Mechanism

### Visualizations
- Equity Curve
- Drawdown Chart
- Returns Distribution
- Feature Importance
- Confusion Matrix
- Trade Analysis Dashboard

## Installation

### Requirements
- Python 3.9+
- pip oder conda

### Setup

```bash
# Clone repository
git clone <repository-url>
cd Trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys (for live trading)
nano .env
```

### Configuration

Bearbeite `config/config.yaml` für deine Trading-Strategie:

```yaml
data:
  symbol: "BTC/USDT"
  timeframes: ["1h"]
  start_date: "2020-01-01"

models:
  algorithms:
    - xgboost
    - lightgbm

backtest:
  initial_capital: 10000
  position_size: 0.02  # 2% per trade
```

## Usage

### 1. Training

Train ein neues Model:

```bash
# Basic training
python train.py --experiment-name my_first_experiment

# Use existing data (skip download)
python train.py --experiment-name test --no-fetch

# Train multiple models and compare
python train.py --experiment-name comparison --multiple-models
```

### 2. Backtesting

Teste ein trainiertes Model:

```bash
# Backtest with specific model
python backtest.py --model data/models/xgboost_20231201_120000.pkl

# Use specific strategy
python backtest.py --model data/models/xgboost_20231201_120000.pkl --strategy momentum_ml

# Custom output directory
python backtest.py --model data/models/xgboost_20231201_120000.pkl --output reports/my_backtest
```

### 3. Live Trading

#### Paper Trading (Empfohlen für Tests)

```bash
python live.py --model data/models/best_model.pkl --mode paper
```

#### Live Trading (ACHTUNG: Echtes Geld!)

```bash
# Requires --confirm flag
python live.py --model data/models/best_model.pkl --mode live --confirm
```

**⚠️ WICHTIG**: Live-Trading ist mit echtem Geld! Teste immer zuerst im Paper-Mode.

## Project Structure

```
Trading/
├── config/
│   └── config.yaml          # Main configuration
├── src/
│   ├── data/                # Data fetching and cleaning
│   ├── features/            # Feature engineering
│   ├── models/              # ML models and training
│   ├── backtest/            # Backtesting engine
│   ├── evaluation/          # Metrics and visualization
│   ├── live_trading/        # Live trading implementation
│   └── utils/               # Utilities
├── data/
│   ├── raw/                 # Raw OHLCV data
│   ├── processed/           # Processed data
│   ├── models/              # Trained models
│   ├── experiments/         # Experiment logs
│   └── logs/                # System logs
├── tests/                   # Unit tests
├── reports/                 # Generated reports
├── notebooks/               # Jupyter notebooks
├── train.py                 # Training script
├── backtest.py              # Backtesting script
├── live.py                  # Live trading script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Architecture

### Data Flow

```
1. Data Fetching (CCXT/YFinance)
   ↓
2. Data Cleaning & Validation
   ↓
3. Feature Engineering (50+ indicators)
   ↓
4. Label Creation (Classification/Regression)
   ↓
5. Train/Val/Test Split (Time-based)
   ↓
6. Model Training + Hyperparameter Tuning
   ↓
7. Backtesting with Risk Management
   ↓
8. Evaluation & Reporting
   ↓
9. Experiment Tracking
   ↓
10. (Optional) Live Deployment
```

### Models

Unterstützte Algorithmen:
- **Logistic Regression**: Baseline Model
- **Random Forest**: Ensemble-basiert
- **XGBoost**: Gradient Boosting (Standard)
- **LightGBM**: Schneller Gradient Boosting
- **LSTM**: (Optional) Deep Learning für Zeitreihen

### Strategies

Verfügbare Trading-Strategien:
- **Pure ML**: Reine ML-Signale ohne Filter
- **Momentum ML**: Kombiniert ML mit Momentum-Indikatoren (SMA Crossover)
- **Mean Reversion ML**: ML + RSI/Bollinger Bands
- **Volatility Breakout ML**: ML + Bollinger Band Breakouts + ADX

### Risk Management

- **Position Sizing**: Feste Fraktion des Kapitals (z.B. 2%)
- **Stop Loss**: Automatischer Stop bei X% Verlust
- **Take Profit**: Automatischer Exit bei X% Gewinn
- **Max Drawdown**: Emergency Stop bei zu großem Drawdown
- **Daily Limits**: Max Trades und Max Loss pro Tag

## Performance Metrics

Das System berechnet folgende Metriken:

### Returns
- **Total Return**: Gesamtrendite
- **CAGR**: Compound Annual Growth Rate
- **Win Rate**: Prozentsatz gewinnender Trades

### Risk-Adjusted
- **Sharpe Ratio**: Return pro Risiko-Einheit
- **Sortino Ratio**: Wie Sharpe, aber nur Downside-Risiko
- **Calmar Ratio**: CAGR / Max Drawdown

### Risk
- **Max Drawdown**: Größter Peak-to-Trough Verlust
- **Volatility**: Standardabweichung der Returns

### Trading
- **Profit Factor**: Gewinn-Trades / Verlust-Trades
- **Average Win/Loss**: Durchschnittlicher Gewinn/Verlust
- **Trade Duration**: Durchschnittliche Haltedauer

## Configuration Guide

### Data Configuration

```yaml
data:
  symbol: "BTC/USDT"           # Trading pair
  exchange: "binance"          # Exchange
  timeframes: ["1h", "4h"]     # Multiple timeframes
  start_date: "2020-01-01"     # Historical data start
```

### Feature Configuration

```yaml
features:
  indicators:
    sma_windows: [10, 20, 50, 100, 200]
    ema_windows: [12, 26, 50]
    rsi_period: 14
    macd: [12, 26, 9]
    bbands_period: 20
    atr_period: 14

  labels:
    type: "classification"      # or "regression"
    horizon: [1, 5, 20]         # Prediction horizons
    threshold: 0.02             # 2% for classification
```

### Model Configuration

```yaml
models:
  test_size: 0.2
  validation_size: 0.1
  cv_splits: 5

  algorithms:
    - logistic_regression
    - random_forest
    - xgboost
    - lightgbm

  hyperparameter_tuning:
    enabled: true
    method: "optuna"            # or "grid_search"
    n_trials: 50
    timeout: 3600               # 1 hour
```

### Backtest Configuration

```yaml
backtest:
  initial_capital: 10000
  position_sizing: "fixed_fraction"
  position_size: 0.02           # 2% per trade

  costs:
    commission: 0.001           # 0.1%
    slippage: 0.0005            # 0.05%

  risk_management:
    stop_loss: 0.02             # 2%
    take_profit: 0.04           # 4%
    max_drawdown: 0.20          # 20%
    max_positions: 1
```

## Testing

Run unit tests:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_backtester.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Experiment Tracking

Alle Experimente werden automatisch geloggt in `data/experiments/experiments.json`.

View experiments:

```python
from src.models.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()

# Get all experiments
df = tracker.get_experiments_df()
print(df)

# Get best experiment
best = tracker.get_best_experiment(metric='sharpe_ratio')
print(best)

# Get leaderboard
leaderboard = tracker.generate_leaderboard(top_n=10)
print(leaderboard)
```

## Safety & Best Practices

### Before Live Trading

1. **Backtest thoroughly**: Min. 1 Jahr historische Daten
2. **Walk-forward analysis**: Teste auf verschiedenen Zeiträumen
3. **Paper trading**: Min. 1 Monat simuliertes Trading
4. **Start small**: Verwende geringe Position Sizes (1-2%)
5. **Monitor constantly**: Überwache Performance täglich

### Risk Management Rules

- **Never risk more than 1-2% per trade**
- **Set stop losses for every trade**
- **Limit daily losses to 5% of capital**
- **Maximum 20% drawdown before emergency stop**
- **Diversify across multiple strategies if possible**

### API Security

- **Never commit API keys to Git**
- **Use read-only keys for fetching data**
- **Enable IP whitelisting on exchange**
- **Use 2FA on exchange account**
- **Store keys in .env file (gitignored)**

## Troubleshooting

### Common Issues

**"Model not found"**
```bash
# Make sure model path is correct
ls data/models/
```

**"API credentials not found"**
```bash
# Check .env file exists and contains keys
cat .env
```

**"CCXT rate limit exceeded"**
```bash
# Increase sleep time in config or reduce data fetching
```

**"Out of memory during training"**
```bash
# Reduce n_trials in hyperparameter tuning
# Or use smaller dataset
```

## Performance Optimization

### Speed up training:
- Use `method: "grid_search"` instead of Optuna for faster (but less optimal) tuning
- Reduce `n_trials` in Optuna
- Use fewer features
- Use `lightgbm` instead of `xgboost` (faster)

### Reduce memory usage:
- Use `parquet` format instead of CSV
- Process data in chunks
- Reduce lookback period
- Use fewer timeframes

## Roadmap

Future improvements:
- [ ] LSTM/GRU neural networks
- [ ] Multi-asset portfolio optimization
- [ ] Reinforcement Learning agents
- [ ] Sentiment analysis integration
- [ ] Advanced order types (limit, trailing stop)
- [ ] Web dashboard for monitoring
- [ ] Telegram/Slack alerts
- [ ] Cloud deployment (AWS/GCP)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Disclaimer

⚠️ **TRADING WARNING** ⚠️

This software is for educational purposes only. Trading cryptocurrencies carries significant risk.

- **You can lose all your money**
- Past performance does not guarantee future results
- Use at your own risk
- The authors are not responsible for any financial losses
- Always test thoroughly before live trading
- Consider consulting a financial advisor

## Support

For issues or questions:
- Open an issue on GitHub
- Check the documentation
- Review existing issues

## Acknowledgments

Built with:
- CCXT for exchange connectivity
- scikit-learn, XGBoost, LightGBM for ML
- Optuna for hyperparameter optimization
- Matplotlib/Seaborn for visualization
- Pandas/NumPy for data processing

---

**Happy Trading! 🚀📈**
