#!/usr/bin/env bash
# ==============================================================================
# UrbanMARL Documentation Build Script
# Builds Sphinx HTML documentation locally and copies .nojekyll for GitHub Pages
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Building UrbanMARL Documentation ==="
cd "${PROJECT_ROOT}"

# Activate virtual environment if present
if [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python"
    SPHINX_BIN=".venv/bin/sphinx-build"
elif command -v sphinx-build &> /dev/null; then
    SPHINX_BIN="sphinx-build"
else
    echo "Error: sphinx-build not found. Install docs extras: pip install -e '.[docs]'"
    exit 1
fi

echo "Using Sphinx binary: ${SPHINX_BIN}"

# Build documentation into docs/_build/html
${SPHINX_BIN} -b html docs/source docs/_build/html

# Ensure .nojekyll is present for GitHub Pages compatibility
touch docs/_build/html/.nojekyll

echo "=== Documentation successfully built! ==="
echo "Output directory: ${PROJECT_ROOT}/docs/_build/html/index.html"
