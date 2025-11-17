# Helper Scripts

Nützliche Scripts für häufige Tasks.

## Available Scripts

### 1. Walk-Forward Analysis

Führe robuste Walk-Forward-Analyse durch:

```bash
python scripts/run_walk_forward.py \
  --data data/raw/BTC_USDT_1h.parquet \
  --model xgboost \
  --strategy pure_ml \
  --output reports/walk_forward
```

**Output:**
- Walk-Forward Plot
- Aggregierte Statistiken
- JSON Results

### 2. Model Comparison

Vergleiche alle trainierten Modelle:

```bash
# Top 10 Modelle
python scripts/compare_models.py --top-n 10

# Sortiert nach Return
python scripts/compare_models.py --metric total_return --top-n 20

# Save to CSV
python scripts/compare_models.py --output reports/leaderboard.csv
```

**Output:**
- Leaderboard Tabelle
- CSV Export
- Summary Statistics

### 3. Strategy Optimization

Optimiere Strategie-Parameter:

```bash
python scripts/optimize_strategy.py \
  --model data/models/xgboost_20231201.pkl \
  --data data/raw/BTC_USDT_1h.parquet \
  --strategy momentum_ml
```

**Optimiert:**
- Position Size
- Stop Loss
- Take Profit

**Output:**
- Beste Parameter
- Performance-Vergleich
- CSV mit allen Kombinationen

### 4. Live Trading Monitor

Überwache Live-Trading in Echtzeit:

```bash
python scripts/monitor_live.py --refresh 5
```

**Zeigt:**
- Aktueller Status (Price, Signal, Position)
- Daily PnL
- Recent History
- Warnings/Errors

Press CTRL+C zum Stoppen.

### 5. Performance Dashboard

Erstelle interaktives Dashboard:

```bash
# Display dashboard
python scripts/dashboard.py

# Save to file
python scripts/dashboard.py --save reports/dashboard.png
```

**Zeigt:**
- Sharpe Ratio Timeline
- Top Experiments
- Returns Distribution
- Sharpe vs Drawdown
- Model Comparison
- Statistics Summary

## Usage Examples

### Complete Analysis Workflow

```bash
# 1. Train multiple models
python train.py --experiment-name production --multiple-models

# 2. Compare models
python scripts/compare_models.py --top-n 10

# 3. Select best model and optimize
python scripts/optimize_strategy.py \
  --model data/models/xgboost_best.pkl \
  --data data/raw/BTC_USDT_1h.parquet

# 4. Walk-forward validation
python scripts/run_walk_forward.py \
  --data data/raw/BTC_USDT_1h.parquet \
  --model xgboost

# 5. Create dashboard
python scripts/dashboard.py --save reports/final_dashboard.png
```

### Live Trading Workflow

```bash
# 1. Start paper trading
python live.py --model data/models/best_model.pkl --mode paper &

# 2. Monitor in another terminal
python scripts/monitor_live.py --refresh 10
```

## Tips

### Walk-Forward Analysis
- Verwende min. 1 Jahr Daten
- train_period: 180 Tage (6 Monate)
- test_period: 30 Tage (1 Monat)
- step: 30 Tage

Konfiguriere in `config/config.yaml`:

```yaml
evaluation:
  walk_forward:
    enabled: true
    train_period: 180
    test_period: 30
    step: 30
```

### Strategy Optimization
- Start mit breiten Ranges
- Narrowe Parameter schrittweise ein
- Achtung vor Overfitting!
- Validiere auf Out-of-Sample Daten

### Monitoring
- Check logs regelmäßig
- Set up alerts für große Drawdowns
- Monitor täglich während Live-Trading
- Review trade history wöchentlich

## Troubleshooting

**"No experiments found"**
```bash
# Run training first
python train.py --experiment-name test
```

**"Data file not found"**
```bash
# Fetch data first
python train.py --experiment-name test  # Auto-fetches data
```

**"Walk-forward takes too long"**
```bash
# Reduce train period or increase step
# Edit config/config.yaml
```

## Advanced Usage

### Batch Processing

```bash
#!/bin/bash
# Run walk-forward for multiple strategies

for strategy in pure_ml momentum_ml mean_reversion_ml
do
  python scripts/run_walk_forward.py \
    --strategy $strategy \
    --output reports/wf_$strategy
done
```

### Automated Reporting

```bash
#!/bin/bash
# Generate daily report

python scripts/compare_models.py --output reports/daily_$(date +%Y%m%d).csv
python scripts/dashboard.py --save reports/dashboard_$(date +%Y%m%d).png
```

## Requirements

All scripts use the same dependencies as the main project.
See `requirements.txt` in project root.
