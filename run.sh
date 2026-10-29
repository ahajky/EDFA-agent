#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${EDFA_ENV_NAME:-edfa-agent}"
ENV_FILE="$ROOT_DIR/environment.yml"
DEFAULT_CONFIG="$ROOT_DIR/configs/three_stage_demo.json"
DEFAULT_WHITEBOX_CONFIG="$ROOT_DIR/configs/whitebox_three_stage_demo.json"
DEFAULT_OUTPUT="$ROOT_DIR/outputs/three_stage_demo"

find_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    printf '%s\n' "$CONDA_EXE"
  elif command -v conda >/dev/null 2>&1; then
    command -v conda
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    printf '%s\n' "$HOME/anaconda3/bin/conda"
  elif [[ -x "/opt/anaconda3/bin/conda" ]]; then
    printf '%s\n' "/opt/anaconda3/bin/conda"
  else
    echo "error: conda was not found" >&2
    echo "set CONDA_EXE to the conda executable and retry" >&2
    exit 1
  fi
}

CONDA_BIN="${CONDA_BIN:-$(find_conda)}"

usage() {
  cat <<'EOF'
Usage: ./run.sh <command> [arguments]

Commands:
  setup                  Create or update the edfa-agent Conda environment
  shell                  Open a Bash shell in the project environment
  doctor                 Show Conda, Python and package versions
  validate [config]      Validate a simulation configuration
  run [config] [output]  Run all configured scenarios
  case <id> [output]     Run one scenario; baseline is still computed
  cases [config]         List scenario IDs
  test [pytest args...]  Run the test suite
  check                  Validate the default config and run tests
  summary [output]       Print the generated cases.csv
  whitebox [config] [--full]
                         Run the white-box rate-propagation model

Examples:
  ./run.sh setup
  ./run.sh run
  ./run.sh case stage2_pump_health_60
  ./run.sh whitebox
  ./run.sh test -vv

Environment variables:
  EDFA_ENV_NAME          Override the Conda environment name (default: edfa-agent)
  CONDA_BIN              Override the Conda executable path
EOF
}

env_exists() {
  "$CONDA_BIN" env list | awk -v name="$ENV_NAME" '
    $1 == name { found = 1 }
    END { exit found ? 0 : 1 }
  '
}

require_env() {
  if ! env_exists; then
    echo "error: Conda environment '$ENV_NAME' does not exist" >&2
    echo "run: ./run.sh setup" >&2
    exit 1
  fi
}

run_in_env() {
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" "$@"
}

require_project() {
  require_env
  if ! run_in_env python -c "import edfa_greybox" >/dev/null 2>&1; then
    echo "error: edfa_greybox is not installed in '$ENV_NAME'" >&2
    echo "run: ./run.sh setup" >&2
    exit 1
  fi
}

setup_env() {
  if env_exists; then
    "$CONDA_BIN" env update -f "$ENV_FILE"
  else
    "$CONDA_BIN" env create -f "$ENV_FILE"
  fi
  run_in_env python -m pip install -e "$ROOT_DIR"
  echo "Environment '$ENV_NAME' is ready."
}

doctor() {
  require_env
  echo "project: $ROOT_DIR"
  echo "conda:   $CONDA_BIN"
  echo "env:     $ENV_NAME"
  run_in_env python - <<'PY'
import platform
import sys

import numpy
import pytest

try:
    import edfa_greybox
except ImportError:
    edfa_greybox_version = "not installed"
else:
    edfa_greybox_version = getattr(edfa_greybox, "__version__", "editable")

print(f"python:   {platform.python_version()} ({sys.executable})")
print(f"numpy:    {numpy.__version__}")
print(f"pytest:   {pytest.__version__}")
print(f"edfa_greybox: {edfa_greybox_version}")
PY
}

command="${1:-help}"
if (($#)); then
  shift
fi

cd "$ROOT_DIR"

case "$command" in
  setup|install)
    setup_env
    ;;
  shell)
    require_env
    exec "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" bash
    ;;
  doctor)
    doctor
    ;;
  validate)
    require_project
    config="${1:-$DEFAULT_CONFIG}"
    run_in_env python -m edfa_greybox.cli validate --config "$config"
    ;;
  run)
    require_project
    config="${1:-$DEFAULT_CONFIG}"
    output="${2:-$DEFAULT_OUTPUT}"
    run_in_env python -m edfa_greybox.cli run --config "$config" --output "$output"
    ;;
  case)
    require_project
    if (($# < 1)); then
      echo "error: case requires a scenario id" >&2
      echo "run './run.sh cases' to list available IDs" >&2
      exit 1
    fi
    case_id="$1"
    output="${2:-$ROOT_DIR/outputs/cases/$case_id}"
    run_in_env python -m edfa_greybox.cli run \
      --config "$DEFAULT_CONFIG" \
      --case "$case_id" \
      --output "$output"
    ;;
  cases|list)
    require_env
    config="${1:-$DEFAULT_CONFIG}"
    run_in_env python - "$config" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
baseline = config.get("baseline_case")
for scenario in config["scenarios"]:
    marker = " (baseline)" if scenario["id"] == baseline else ""
    print(f"{scenario['id']}{marker}")
PY
    ;;
  test)
    require_project
    run_in_env python -m pytest "$@"
    ;;
  check)
    require_project
    run_in_env python -m edfa_greybox.cli validate --config "$DEFAULT_CONFIG"
    run_in_env python -m pytest
    ;;
  whitebox)
    require_project
    config="$DEFAULT_WHITEBOX_CONFIG"
    if (($#)) && [[ "$1" != --* ]]; then
      config="$1"
      shift
    fi
    run_in_env python -m edfa_whitebox "$config" "$@"
    ;;
  summary)
    output="${1:-$DEFAULT_OUTPUT}"
    if [[ ! -f "$output/cases.csv" ]]; then
      echo "error: $output/cases.csv does not exist" >&2
      echo "run: ./run.sh run" >&2
      exit 1
    fi
    cat "$output/cases.csv"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "error: unknown command '$command'" >&2
    usage >&2
    exit 1
    ;;
esac
