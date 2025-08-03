# Makefile for Data Preprocessing Research

.PHONY: help setup install test clean baseline preprocessing analysis all

# Default target
help:
	@echo "Data Preprocessing Research Framework"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup          - Setup the development environment"
	@echo "  install        - Install dependencies"
	@echo "  test           - Run unit tests"
	@echo "  clean          - Clean generated files and cache"
	@echo "  baseline       - Run baseline experiments"
	@echo "  preprocessing  - Run preprocessing experiments"
	@echo "  analysis       - Generate analysis notebooks"
	@echo "  all            - Run complete experimental pipeline"
	@echo ""

# Setup development environment
setup:
	@echo "Setting up development environment..."
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully"

# Run unit tests
test:
	@echo "Running unit tests..."
	python -m pytest tests/ -v --cov=src --cov-report=html
	@echo "Tests completed. Coverage report available in htmlcov/"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/*/__pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf experiments/results/*/
	rm -rf datasets/processed/*
	@echo "Cleanup completed"

# Run baseline experiments
baseline:
	@echo "Running baseline experiments..."
	python experiments/scripts/run_baseline.py
	@echo "Baseline experiments completed"

# Run preprocessing experiments
preprocessing:
	@echo "Running preprocessing experiments..."
	python experiments/scripts/run_preprocessing_study.py
	@echo "Preprocessing experiments completed"

# Generate analysis notebooks
analysis:
	@echo "Generating analysis notebooks..."
	jupyter nbconvert --execute notebooks/results_analysis/*.ipynb
	@echo "Analysis notebooks generated"

# Run complete pipeline
all: clean baseline preprocessing analysis
	@echo "Complete experimental pipeline completed"

# Development helpers
format:
	@echo "Formatting code..."
	black src/ tests/ experiments/
	isort src/ tests/ experiments/
	@echo "Code formatting completed"

lint:
	@echo "Running linting..."
	flake8 src/ tests/ experiments/
	@echo "Linting completed"

# Docker support
docker-build:
	@echo "Building Docker image..."
	docker build -t preprocessing-research .
	@echo "Docker image built successfully"

docker-run:
	@echo "Running experiments in Docker..."
	docker run -v $(PWD)/experiments/results:/app/experiments/results preprocessing-research
	@echo "Docker experiments completed"
