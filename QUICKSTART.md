# Quick Start Guide

Schnelleinstieg in 5 Minuten!

## 1. Installation (2 Minuten)

```bash
# Clone und Setup
git clone <repository-url>
cd Trading
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## 2. Erstes Training (1 Minute)

```bash
# Train ein XGBoost Model auf BTC/USDT
python train.py --experiment-name my_first_test
```

Das Skript wird:
- ✓ Daten von Binance herunterladen
- ✓ Features generieren (50+ Indikatoren)
- ✓ Model trainieren
- ✓ Backtest durchführen
- ✓ Report generieren

**Output**: `reports/my_first_test_<timestamp>/`

## 3. Ergebnisse ansehen (1 Minute)

```bash
# Siehe Text-Report
cat reports/my_first_test_*/report.txt

# Siehe Visualisierungen
ls reports/my_first_test_*/
# equity_curve.png
# drawdown.png
# feature_importance.png
# returns_distribution.png
```

## 4. Backtest mit trainiertem Model (30 Sekunden)

```bash
# Finde dein Model
ls data/models/

# Run Backtest
python backtest.py --model data/models/xgboost_*.pkl --strategy pure_ml
```

## 5. Paper Trading testen (30 Sekunden)

```bash
# Starte Paper Trading (Simulation)
python live.py --model data/models/xgboost_*.pkl --mode paper
```

**CTRL+C** zum Stoppen

## Nächste Schritte

### Konfiguration anpassen

Editiere `config/config.yaml`:

```yaml
data:
  symbol: "ETH/USDT"  # Ändere auf ETH

backtest:
  position_size: 0.05  # Erhöhe auf 5%
```

### Verschiedene Strategien testen

```bash
# Momentum Strategy
python backtest.py --model data/models/xgboost_*.pkl --strategy momentum_ml

# Mean Reversion Strategy
python backtest.py --model data/models/xgboost_*.pkl --strategy mean_reversion_ml

# Volatility Breakout Strategy
python backtest.py --model data/models/xgboost_*.pkl --strategy volatility_breakout_ml
```

### Multiple Modelle vergleichen

```bash
python train.py --experiment-name comparison --multiple-models
```

Trainiert alle konfigurierten Modelle und vergleicht Performance.

### Experimente ansehen

```python
from src.models.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()
print(tracker.generate_leaderboard())
```

## Typische Workflows

### Workflow 1: Entwicklung & Testing

```bash
# 1. Train model
python train.py --experiment-name dev_test

# 2. Backtest
python backtest.py --model data/models/xgboost_*.pkl

# 3. Paper trade für 1 Tag
python live.py --model data/models/xgboost_*.pkl --mode paper
```

### Workflow 2: Produktion

```bash
# 1. Train mit voller Historie
python train.py --experiment-name production_v1

# 2. Umfangreicher Backtest
python backtest.py --model data/models/xgboost_*.pkl --output reports/production_backtest

# 3. Paper trading für 1 Woche
python live.py --model data/models/xgboost_*.pkl --mode paper

# 4. (Optional) Live trading
# Siehe README für Safety Guidelines!
```

## Häufige Anfänger-Fehler

### ❌ Zu wenig Daten
```yaml
# Schlecht: Nur 1 Monat
start_date: "2023-11-01"

# Gut: Min. 1 Jahr
start_date: "2022-01-01"
```

### ❌ Overfitting
```yaml
# Schlecht: Zu komplexe Modelle auf wenig Daten
hyperparameter_tuning:
  n_trials: 200  # Zu viel!

# Gut: Moderate Komplexität
hyperparameter_tuning:
  n_trials: 50
```

### ❌ Zu große Position Sizes
```yaml
# Schlecht: 50% pro Trade = hohes Risiko
position_size: 0.5

# Gut: 1-2% pro Trade
position_size: 0.02
```

### ❌ Kein Risk Management
```yaml
# Schlecht: Keine Stops
risk_management:
  stop_loss: null
  max_drawdown: null

# Gut: Klare Limits
risk_management:
  stop_loss: 0.02
  take_profit: 0.04
  max_drawdown: 0.20
```

## Hilfe & Support

- **Dokumentation**: Siehe `README.md`
- **Tests**: `pytest tests/ -v`
- **Logs**: `data/logs/`
- **Experimente**: `data/experiments/experiments.json`

## Performance Benchmarks

Typische Ergebnisse (BTC/USDT, 2020-2023):

| Model | Sharpe Ratio | Max DD | Win Rate |
|-------|-------------|---------|----------|
| XGBoost | 1.2 - 2.0 | 15-25% | 55-65% |
| LightGBM | 1.0 - 1.8 | 18-28% | 52-62% |
| Random Forest | 0.8 - 1.5 | 20-30% | 50-60% |

*Disclaimer: Past performance ≠ Future results*

## Troubleshooting

**"No module named 'src'"**
```bash
# Run from project root
cd /path/to/Trading
python train.py
```

**"Data download fails"**
```bash
# Use existing data
python train.py --no-fetch
```

**"Out of memory"**
```bash
# Reduce trials
# Edit config.yaml: n_trials: 20
```

---

**Ready to go! 🚀**

Für mehr Details: `README.md`
