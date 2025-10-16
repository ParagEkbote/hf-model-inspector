# --------------------------------------------------------
# Makefile
# Automates linting, testing, and type-checking
# --------------------------------------------------------

.PHONY: all lint format check fix typecheck precommit test unit integration quick mutate clean help

# Default target
all: check test

# --------------------------------------------------------
# 🔍 Linting and formatting
# --------------------------------------------------------

# Check all linting without fixing
check:
	@echo "Running all linting checks..."
	@hatch run lint:check

# Auto-fix formatting issues
fix:
	@echo "Auto-fixing formatting..."
	@hatch run lint:fix

# Individual linting tools
black-check:
	@hatch run lint:black-check

black-fix:
	@hatch run lint:black-fix

flake8:
	@hatch run lint:flake

isort-check:
	@hatch run lint:isort-check

isort-fix:
	@hatch run lint:isort-fix

# Alias for check
lint: check

# Alias for fix
format: fix

# --------------------------------------------------------
# Type checking
# --------------------------------------------------------
typecheck:
	@echo "Running type check..."
	@hatch run lint:typecheck

ty: typecheck

# --------------------------------------------------------
# Pre-commit hooks
# --------------------------------------------------------
precommit:
	@echo "Running pre-commit hooks..."
	@hatch run lint:precommit

# --------------------------------------------------------
# Testing
# --------------------------------------------------------

# Run all tests with coverage
test:
	@echo "Running all tests..."
	@hatch run test:all

# Unit tests only
unit:
	@echo "Running unit tests..."
	@hatch run test:unit

# Integration tests only
integration:
	@echo "Running integration tests..."
	@hatch run test:integration

# Quick tests (no coverage)
quick:
	@echo "Running quick tests..."
	@hatch run test:quick

# Mutation testing
mutate:
	@echo "Running mutation tests..."
	@hatch run mutation:run

mutate-results:
	@hatch run mutation:results

mutate-html:
	@hatch run mutation:html

# --------------------------------------------------------
# CI/CD
# --------------------------------------------------------

# Full CI pipeline
ci: check typecheck test
	@echo "✓ CI pipeline completed successfully"

# --------------------------------------------------------
# Package management
# --------------------------------------------------------

# Build package
build:
	@echo "Building package..."
	@hatch run publish:build

# Check package before upload
check-package:
	@hatch run publish:check-dist

# Upload to PyPI
upload:
	@hatch run publish:upload

# --------------------------------------------------------
# Cleanup
# --------------------------------------------------------
clean:
	@echo "Cleaning up..."
	@rm -rf .pytest_cache .mutmut-cache .coverage coverage.xml htmlcov
	@rm -rf dist build *.egg-info
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete

# Deep clean (including hatch environments)
clean-all: clean
	@echo "Deep cleaning (including hatch environments)..."
	@hatch env prune

# --------------------------------------------------------
# Help
# --------------------------------------------------------
help:
	@echo "Available targets:"
	@echo ""
	@echo "Linting & Formatting:"
	@echo "  check          - Run all linting checks without fixing"
	@echo "  fix            - Auto-fix formatting issues (black, isort)"
	@echo "  lint           - Alias for check"
	@echo "  format         - Alias for fix"
	@echo "  black-check    - Check Black formatting only"
	@echo "  black-fix      - Fix Black formatting only"
	@echo "  flake8         - Run Flake8 linter only"
	@echo "  isort-check    - Check isort formatting only"
	@echo "  isort-fix      - Fix isort formatting only"
	@echo ""
	@echo "Type Checking:"
	@echo "  typecheck      - Run Ty type checker"
	@echo "  ty             - Alias for typecheck"
	@echo ""
	@echo "Pre-commit:"
	@echo "  precommit      - Run all pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests with coverage"
	@echo "  unit           - Run unit tests only"
	@echo "  integration    - Run integration tests only"
	@echo "  quick          - Run tests without coverage"
	@echo "  mutate         - Run mutation testing"
	@echo "  mutate-results - Show mutation test results"
	@echo "  mutate-html    - Generate HTML mutation report"
	@echo ""
	@echo "CI/CD:"
	@echo "  ci             - Full CI pipeline (check + typecheck + test)"
	@echo ""
	@echo "Package Management:"
	@echo "  build          - Build the package"
	@echo "  check-package  - Verify package before upload"
	@echo "  upload         - Upload package to PyPI"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          - Remove build artifacts and cache"
	@echo "  clean-all      - Deep clean including hatch environments"
	@echo ""
	@echo "Default target: all (runs check + test)"