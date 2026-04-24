# Navigator (ino) - Makefile

.PHONY: run test test-unit test-integration install install-pi lint clean health smoke scrape

## Run the robot
run:
	python -m app.main

## Run all tests
test:
	pytest tests/ -q

## Run unit tests only
test-unit:
	pytest tests/unit/ -q

## Run integration tests only
test-integration:
	pytest tests/integration/ -q

## Run smoke test (mock mode, no hardware needed)
smoke:
	python scripts/smoke_test.py

## Run health check
health:
	python scripts/health_check.py

## Install dependencies (dev / x86)
install:
	pip install --upgrade pip
	pip install -r requirements.txt

## Install dependencies for Raspberry Pi 5 (CPU-only torch)
install-pi:
	pip install --upgrade pip
	pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

## Check syntax
lint:
	python -m py_compile app/main.py && echo "Syntax OK"
	python -m py_compile app/pipeline/controller.py && echo "Controller OK"

## Scrape ECU website (refreshes local knowledge cache)
scrape:
	python scripts/scrape_ecu.py
	python scripts/scrape_ecu_arabic.py

## Clean Python cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	echo "Clean done"
