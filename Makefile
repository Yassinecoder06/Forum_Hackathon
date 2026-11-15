.PHONY: all setup dev dev-backend dev-frontend test test-schemas test-backend test-experiments test-frontend lint smoke clean

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON ?= $(VENV_DIR)/bin/python3
API_URL ?= http://127.0.0.1:8000

all: test

setup:
	@bash scripts/setup.sh

dev:
	@bash scripts/dev.sh

dev-backend:
	@$(VENV_PYTHON) -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

dev-frontend:
	@npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173 --strictPort

lint:
	@npm --prefix frontend run lint

test:
	@bash scripts/test.sh

test-schemas:
	@$(VENV_PYTHON) -m pytest backend/app/schemas -v

test-backend:
	@$(VENV_PYTHON) -m pytest backend/tests -v

test-experiments:
	@$(VENV_PYTHON) -m pytest experiments/motion_extraction -v

test-frontend:
	@npm --prefix frontend run lint
	@npm --prefix frontend run build

smoke:
	@$(VENV_PYTHON) scripts/smoke_test.py --base-url $(API_URL)

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@rm -rf .pytest_cache
