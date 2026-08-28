#!/usr/bin/env bash
#
# Formats, lints, type-checks, and tests this project — the same
# black / flake8 / mypy / pytest sequence that's been run by hand after
# every feature. One command instead of chaining them, and easy to
# audit (this file *is* the audit trail) or copy into another project
# to standardize the same checks there.
#
# Usage:
#   ./scripts/check.sh
#
# Runs every step even if an earlier one fails, so one run gives a full
# picture rather than stopping at the first problem. Exits non-zero if
# any step failed.
#
# Requires the dev tools in requirements-dev.txt:
#   pip install -r requirements-dev.txt

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SRC_DIR="src"
TEST_DIR="tests"
MAX_LINE_LENGTH=100

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
    black "$SRC_DIR" "$TEST_DIR"

run_step "flake8 (lint)" \
    flake8 --max-line-length="$MAX_LINE_LENGTH" "$SRC_DIR" "$TEST_DIR"

run_step "mypy (type-check)" \
    mypy --ignore-missing-imports "$SRC_DIR"

run_step "pytest (tests)" \
    python3 -m pytest "$TEST_DIR" -q

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
