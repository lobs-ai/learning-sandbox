.PHONY: setup run run-headless test clean help

VENV := .venv
PYTHON := $(VENV)/bin/python3

help:
	@echo "Obstacle Runner - 3D RL Agent"
	@echo ""
	@echo "  make setup         Install dependencies"
	@echo "  make run           Run with GUI"
	@echo "  make run-headless  Run headless (fast training)"
	@echo "  make test          Run headless for 50 episodes"
	@echo "  make train-short   Run headless for 200 episodes"
	@echo "  make clean         Remove venv"

setup:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV); \
	fi
	@echo "Installing dependencies..."
	@$(PYTHON) -m pip install pyglet torch numpy --quiet
	@echo "Done! Run 'make run' to start."

run:
	$(PYTHON) obstacle_runner.py

run-headless:
	$(PYTHON) obstacle_runner.py --headless

test:
	$(PYTHON) obstacle_runner.py --headless 50

clean:
	rm -rf $(VENV)

train-short:
	$(PYTHON) obstacle_runner.py --headless 200