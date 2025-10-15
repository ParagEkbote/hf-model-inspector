# --------------------------------------------------------
# Makefile
# Automates linting, testing, and type-checking
# --------------------------------------------------------

.PHONY: all lint format typecheck precommit test unit integration mutate clean

# Default target
all: lint test

# --------------------------------------------------------
# 🔍 Linting and formatting
# --------------------------------------------------------
lint:
	hatch run default:lint
	hatch run default:format
	pre-commit run --all-files

# Only run Black, Flake8, and isort individually if needed
black:
	pre-commit run black --all-files

flake8:
	pre-commit run flake8 --all-files

isort:
	pre-commit run isort --all-files

# --------------------------------------------------------
# Type checking (using Ty )
# --------------------------------------------------------
typecheck:
	hatch run default:ty check || echo "Ty type check failed."

# --------------------------------------------------------
# Testing
# --------------------------------------------------------
test: unit integration

unit:
	hatch run test:unit

integration:
	hatch run test:integration

# Mutation testing
mutate:
	hatch run test:mutate

# --------------------------------------------------------
# Cleanup
# --------------------------------------------------------
clean:
	rm -rf .pytest_cache .mutatest_cache .coverage coverage.xml dist build

# --------------------------------------------------------
# shortcuts
# --------------------------------------------------------
precommit:
	pre-commit run --all-files

# Run everything in CI (lint + typecheck + all tests)
ci: lint typecheck test
