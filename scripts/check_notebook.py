"""Execute energy_forecast.ipynb end to end and guard its headline metrics.

CI has two things worth catching in this repo: the notebook no longer runs at
all, and a change that quietly wrecks the model while still running clean. So
this executes every cell in order, then re-reads the metrics the notebook itself
printed and compares them against a floor.

The thresholds are deliberately loose around the README baseline (RMSE 3,774 /
MAE 2,925 / MAPE 9.31% / R2 0.658). XGBoost is seeded and reproducible on a
fixed platform, but floating-point results drift slightly between macOS/arm64
and the Linux/x86-64 runner, so exact-match assertions would be flaky. These
bounds absorb that noise while still failing on a genuine regression -- a
dropped feature, a broken split, a misaligned lag.

Run locally the same way CI does:

    uv run python scripts/check_notebook.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "energy_forecast.ipynb"
OUTPUT_DIR = REPO_ROOT / "build"
EXECUTED_NOTEBOOK = OUTPUT_DIR / "energy_forecast.executed.ipynb"

CELL_TIMEOUT_SECONDS = 900

# name -> (regex over the notebook's stdout, comparison, bound)
# "max" means the value must be <= bound; "min" means it must be >= bound.
METRIC_CHECKS: list[tuple[str, str, str, float]] = [
    ("RMSE", r"Test RMSE:\s*([\d,\.]+)", "max", 4200.0),
    ("MAE", r"Test MAE:\s*([\d,\.]+)", "max", 3300.0),
    ("MAPE", r"Test MAPE:\s*([\d\.]+)%", "max", 10.5),
    ("R2", r"Test R.:\s*([\d\.]+)", "min", 0.60),
]


def execute_notebook() -> nbformat.NotebookNode:
    """Run every cell from the repo root, so the notebook finds PJME_hourly.csv."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=CELL_TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )

    started = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as error:
        print(f"::error::Notebook failed to execute: {error}", file=sys.stderr)
        raise
    finally:
        # Save even on failure -- the executed copy is uploaded as a CI artifact
        # and is the fastest way to see which cell blew up and why.
        OUTPUT_DIR.mkdir(exist_ok=True)
        nbformat.write(notebook, EXECUTED_NOTEBOOK)

    print(f"Notebook executed in {time.perf_counter() - started:.1f}s")
    return notebook


def collect_stdout(notebook: nbformat.NotebookNode) -> str:
    """Concatenate every stream output, which is where the metrics are printed."""
    return "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def check_metrics(stdout: str) -> list[str]:
    failures: list[str] = []

    for name, pattern, comparison, bound in METRIC_CHECKS:
        match = re.search(pattern, stdout)
        if match is None:
            failures.append(
                f"{name}: could not find it in the notebook output. "
                f"If the notebook stopped printing it, update METRIC_CHECKS."
            )
            continue

        value = float(match.group(1).replace(",", ""))
        breached = value > bound if comparison == "max" else value < bound
        limit = f"{'<=' if comparison == 'max' else '>='} {bound:,.2f}"

        if breached:
            failures.append(f"{name}: {value:,.3f} breaches the bound ({limit})")
        else:
            print(f"  OK  {name:<5} {value:>10,.3f}  (bound {limit})")

    return failures


def main() -> int:
    notebook = execute_notebook()
    failures = check_metrics(collect_stdout(notebook))

    if failures:
        print("\nModel metrics regressed:", file=sys.stderr)
        for failure in failures:
            print(f"::error::{failure}", file=sys.stderr)
        return 1

    print("\nNotebook ran clean and all metrics are within bounds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
