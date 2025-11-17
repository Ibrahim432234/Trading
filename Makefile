# Makefile for ML Trading System

.PHONY: help install test train backtest live clean format lint

help:
	@echo "ML Trading System - Available Commands:"
	@echo ""
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run unit tests"
	@echo "  make train         - Train default model"
	@echo "  make backtest      - Run backtest"
	@echo "  make live-paper    - Start paper trading"
	@echo "  make dashboard     - Show performance dashboard"
	@echo "  make clean         - Clean generated files"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with flake8"
	@echo ""

install:
	pip install -r requirements.txt
	cp .env.example .env
	@echo "Installation complete! Edit .env with your API keys."

test:
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Tests complete. Coverage report: htmlcov/index.html"

train:
	python train.py --experiment-name default_experiment

train-multi:
	python train.py --experiment-name multi_model --multiple-models

backtest:
	@echo "Finding latest model..."
	@MODEL=$$(ls -t data/models/*.pkl 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ]; then \
		echo "No model found. Run 'make train' first."; \
		exit 1; \
	fi; \
	python backtest.py --model $$MODEL

live-paper:
	@MODEL=$$(ls -t data/models/*.pkl 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ]; then \
		echo "No model found. Run 'make train' first."; \
		exit 1; \
	fi; \
	python live.py --model $$MODEL --mode paper

walk-forward:
	python scripts/run_walk_forward.py --data data/raw/*.parquet

compare:
	python scripts/compare_models.py --top-n 10

dashboard:
	python scripts/dashboard.py

optimize:
	@MODEL=$$(ls -t data/models/*.pkl 2>/dev/null | head -1); \
	DATA=$$(ls -t data/raw/*.parquet 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ] || [ -z "$$DATA" ]; then \
		echo "Model or data not found."; \
		exit 1; \
	fi; \
	python scripts/optimize_strategy.py --model $$MODEL --data $$DATA

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	@echo "Cleaned cache files"

format:
	black src/ tests/ *.py scripts/
	@echo "Code formatted with black"

lint:
	flake8 src/ tests/ *.py scripts/ --max-line-length=100 --ignore=E501,W503
	@echo "Linting complete"

fetch-data:
	@echo "Fetching BTC/USDT data..."
	python -c "from src.data.data_fetcher import DataFetcher; \
	           from src.data.data_cleaner import DataCleaner; \
	           f = DataFetcher('binance'); \
	           df = f.fetch_ohlcv('BTC/USDT', '1h', '2023-01-01'); \
	           c = DataCleaner(); \
	           df = c.clean_ohlcv(df); \
	           df.to_parquet('data/raw/BTC_USDT_1h.parquet'); \
	           print(f'Saved {len(df)} candles')"

setup: install fetch-data
	@echo "Setup complete! Run 'make train' to train your first model."

full-pipeline: clean fetch-data train backtest dashboard
	@echo "Full pipeline complete!"

monitor:
	python scripts/monitor_live.py

.DEFAULT_GOAL := help
