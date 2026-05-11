"""Regression test for mend2np task scoring.

Re-runs every example driver against the bundled example CSVs, then compares
the resulting score/trial CSVs against frozen baselines in `tests/expected/`.

Run directly:        python tests/test_regression.py
Or via pytest:       pytest tests/test_regression.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
OUT_DIR = HERE / 'out'
EXPECTED_DIR = HERE / 'expected'

# Each driver script and the (task_prefix, output_glob_pattern) pairs it produces.
# The glob matches against the timestamped filename written by `write_out`.
DRIVERS = {
    'sert':     ('example_driver_sert.py',     [('sert_scores',     'sert_*_scores_*.csv'),
                                                 ('sert_trials',     'sert_*_trials_*.csv')]),
    'pgng':     ('example_driver_pgng.py',     [('pgng_scores',     'PGNGS_*_scores_*.csv'),
                                                 ('pgng_trials',     'PGNGS_*_trials_*.csv')]),
    'bart':     ('example_driver_bart.py',     [('bart_scores',     'BART_*_scores_*.csv'),
                                                 ('bart_trials',     'BART_*_trials_*.csv')]),
    'fept':     ('example_driver_fept.py',     [('fept_scores',     'FEPT_*_scores_*.csv')]),
    'synonyms': ('example_driver_synonyms.py', [('synonyms_scores', 'Synonyms_*_scores_*.csv'),
                                                 ('synonyms_trials', 'Synonyms_*_trials_*.csv')]),
}


def _clear_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.iterdir():
        if p.is_file():
            p.unlink()


def _run_driver(driver_filename: str) -> None:
    driver = HERE / driver_filename
    result = subprocess.run(
        [sys.executable, str(driver)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'driver {driver_filename} exited {result.returncode}\n'
            f'STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}'
        )


def _find_one(pattern: str) -> Path:
    matches = sorted(OUT_DIR.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f'no output matched {pattern} in {OUT_DIR}')
    if len(matches) > 1:
        raise RuntimeError(f'expected one output for {pattern}, got {len(matches)}: {matches}')
    return matches[0]


def _compare(actual: Path, expected: Path) -> None:
    actual_df = pd.read_csv(actual)
    expected_df = pd.read_csv(expected)
    # Columns must match exactly (in name and order).
    if list(actual_df.columns) != list(expected_df.columns):
        only_actual = set(actual_df.columns) - set(expected_df.columns)
        only_expected = set(expected_df.columns) - set(actual_df.columns)
        raise AssertionError(
            f'column mismatch for {actual.name} vs {expected.name}\n'
            f'  in actual only: {sorted(only_actual)}\n'
            f'  in expected only: {sorted(only_expected)}'
        )
    # Use pandas testing for tolerant numeric comparison + clear diff messages.
    try:
        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9)
    except AssertionError as e:
        raise AssertionError(f'content mismatch for {actual.name} vs {expected.name}:\n{e}') from None


def _check_task(task: str) -> None:
    driver_filename, pairs = DRIVERS[task]
    _clear_out_dir()
    _run_driver(driver_filename)
    for stable_name, pattern in pairs:
        actual = _find_one(pattern)
        expected = EXPECTED_DIR / f'{stable_name}.csv'
        if not expected.exists():
            raise FileNotFoundError(f'baseline missing: {expected}')
        _compare(actual, expected)


def test_sert():     _check_task('sert')
def test_pgng():     _check_task('pgng')
def test_bart():     _check_task('bart')
def test_fept():     _check_task('fept')
def test_synonyms(): _check_task('synonyms')


def main() -> int:
    failures: list[str] = []
    for task in DRIVERS:
        try:
            _check_task(task)
            print(f'  PASS  {task}')
        except Exception as e:
            print(f'  FAIL  {task}: {e}')
            failures.append(task)
    print()
    print(f'{len(DRIVERS) - len(failures)}/{len(DRIVERS)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
