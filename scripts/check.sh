#!/usr/bin/env bash
#
# Formats, lints, type-checks, and tests this project — the same
# black / flake8 / mypy / pytest sequence that's been run by hand after
# every feature. One command instead of chaining them, and easy to
# audit (this file *is* the audit trail) or copy into another project
# to standardize the same checks there.
#
# Usage:
#   ./scripts/check.sh              # skips tests marked live_data
#   ./scripts/check.sh --live-data  # also runs them
#
# A live_data test (see tests/conftest.py) reads real project data under
# data/ rather than a fixture - it checks the data, not the code, and
# can be slow - so it only runs when asked.
#
# pytest runs on $PYTEST_WORKERS processes (pytest-xdist), default 4:
# on a 10-core Mac, 4 was faster than 8 or one per core. Set it to 0 to
# run in a single process.
#
# Runs every step even if an earlier one fails, so one run gives a full
# picture rather than stopping at the first problem. Exits non-zero if
# any step failed.
#
# Every tool runs as `$PYTHON -m <tool>`, so all four steps share one
# interpreter and its packages. $PYTHON defaults to the project venv
# (venv/bin/python, or venv/Scripts/python.exe on Windows), then
# python3 on PATH. Override it to use another environment, e.g. a ROCm
# venv:
#   PYTHON=/path/to/rocm-venv/bin/python ./scripts/check.sh
#
# That interpreter needs requirements.txt and the dev tools in
# requirements-dev.txt:
#   $PYTHON -m pip install -r requirements.txt -r requirements-dev.txt

set -uo pipefail

PYTEST_MARKERS=(-m "not live_data")
for arg in "$@"; do
    case "$arg" in
        --live-data) PYTEST_MARKERS=() ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${PYTHON:-}" ]; then
    if [ -x venv/bin/python ]; then
        PYTHON="venv/bin/python"
    elif [ -x venv/Scripts/python.exe ]; then
        PYTHON="venv/Scripts/python.exe"
    else
        PYTHON="python3"
    fi
fi
echo "Using $PYTHON ($("$PYTHON" --version 2>&1))"
echo

SRC_DIR="src"
TEST_DIR="tests"

failed=0

run_step() {
    local name="$1"
    shift
    echo "-- ${name} --"
    if "$@"; then
        echo "PASS: ${name}"
    else
        echo "FAIL: ${name}"
        failed=1
    fi
    echo
}

run_step "black (auto-format)" \
    "$PYTHON" -m black "$SRC_DIR" "$TEST_DIR"

run_step "flake8 (lint)" \
    "$PYTHON" -m flake8 "$SRC_DIR" "$TEST_DIR"

run_step "mypy (type-check)" \
    "$PYTHON" -m mypy --ignore-missing-imports "$SRC_DIR"

run_step "pytest (tests)" \
    "$PYTHON" -m pytest "$TEST_DIR" -q -n "${PYTEST_WORKERS:-4}" \
    ${PYTEST_MARKERS[@]+"${PYTEST_MARKERS[@]}"}

echo "-- cleanup --"
rm -rf .pytest_cache
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "removed .pytest_cache and __pycache__ directories"
echo

if [ "$failed" -eq 0 ]; then
    echo "All checks passed."
else
    echo "One or more checks failed — see output above."
fi

exit "$failed"
