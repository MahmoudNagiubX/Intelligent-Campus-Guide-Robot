# Navigator - Makefile

.PHONY: run test install lint clean

## Install dependencies
install:
	pip install -r requirements.txt

## Run the application
run:
	python -m app.main

## Run all tests
test:
	pytest tests/ -v

## Run only unit tests
test-unit:
	pytest tests/unit/ -v

## Run only integration tests
test-integration:
	pytest tests/integration/ -v

## Lint
lint:
	python -m py_compile app/main.py && echo "Syntax OK ✅"

## Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	echo "Clean ✅"
