# mend2_neuropsych (`mend2np`)

Python scoring scripts for behavioural CSV data from neuropsych tasks run in the MEND2 lab. Reads PsychoPy / E-Prime CSV output, applies experiment-version-specific column mappings via a JSON config, and writes aggregated trial-level and score-level CSVs.

Eight tasks are supported:

| Task | What it measures | Example config |
| --- | --- | --- |
| **sert** | Suicide Emotion Rigidity Task; per-cue accuracy and switch cost RTs | [`tests/sert_example.json`](tests/sert_example.json) |
| **pgng** | Parametric Go / No-go / Stop; hit / miss / commission counts and RTs per block | [`tests/pgng_example.json`](tests/pgng_example.json) |
| **bart** | Balloon Analogue Risk Task; pumps, pops, earnings, post-failure caution | [`tests/bart_example.json`](tests/bart_example.json) |
| **fept** | Facial Emotion Perception Task; per-emotion / race / sex / animal accuracy and misclassification counts | [`tests/fept_example.json`](tests/fept_example.json) |
| **synonyms** | Synonym matching; accuracy and RT by correctness | [`tests/synonyms_example.json`](tests/synonyms_example.json) |
| **fingosc** | Finger Oscillation; mean and SD of tap RT per block (dominant vs non-dominant hand) | [`tests/fingosc_example.json`](tests/fingosc_example.json) (keyboard); also `fingosc_example_touch.json` and `fingosc_example_stacked.json` |
| **smid** | Social Monetary Incentive Delay; per-condition (self / charity × gain / lose / neither × small / big) probe-response counts and RTs, with the participant's charity & rep carried through | [`tests/smid_example.json`](tests/smid_example.json) |
| **stroop** | Classic + emotional Stroop (alternating blocks); per `test × condition` n-trials / accuracy / RTs, plus Stroop-interference contrast scores | [`tests/stroop_example.json`](tests/stroop_example.json) (keyboard); also `stroop_example_touch.json` |

## Requirements

- Python ≥ 3.12
- `pandas`, `numpy`, `easygui` (installed as dependencies)

## Install

```bash
pip install git+https://github.com/MEND2-Lab/mend2np.git
```

## How to run

There are two ways to use mend2np — pick whichever matches your comfort level. Both produce identical output.

### Option A: edit a JSON config, run the driver script (no Python knowledge required)

1. Open the matching example config in `tests/` (e.g. `sert_example.json`) and copy it into your own working directory.
2. Edit the config so the right-hand side of every `"standard_name": "csv_column_name"` pair matches the columns in your CSV. See [`docs/CONFIGURING.md`](docs/CONFIGURING.md) for a step-by-step walkthrough.
3. Open the matching driver script (e.g. `tests/example_driver_sert.py`), change `filelist=[...]` to point at your CSV(s), and run it: `python my_driver_sert.py`.
4. Look in the `out/` directory for `<exp>_n<N>_scores_<timestamp>.csv` and `<exp>_n<N>_trials_<timestamp>.csv`.

### Option B: call the function from Python

```python
from mend2np import sert

params = { ... }  # same dict shape as the JSON
sert(
    params=params,
    filelist=['data/subject1.csv', 'data/subject2.csv'],
    out='out',
    write=True,
)
```

## Common arguments

These are accepted by every task function. Defaults are sensible for first-time use.

| Argument | Type | Default | What it does |
| --- | --- | --- | --- |
| `params` | dict | required | Maps your experiment's CSV column names to the standard names this library uses. Load from JSON or build in Python. |
| `filelist` | list/str | `''` | A list of CSV paths to score, OR the path to a text file with one CSV path per line. Leave blank to pop up a GUI file picker. |
| `out` | str | current dir | Directory to write outputs to. Created if it doesn't exist. |
| `write` | bool | `True` | If False, results are returned but not written. |
| `formatted` | bool | `False` | True if your CSV is already "tidy" with the library's standard column names. Most users want `False`. |
| `log` | int/str | `20` (INFO) | Log level. Use `10` for very verbose debug output, `30` for warnings only. |
| `trial_filter` | str | `''` | Optional pandas query string to subset trials before scoring (e.g. `'event_type == "go"'`). |

`pgng` adds `ind` (write a per-file TSV) and `platform` (`'psychopy'` or `'eprime'`).
`fept` adds `ind` (write per-file CSV) but has no `trial_filter`.

## Output files

Each run writes (at most) two CSVs into `out/`:

- `<exp_name>_n<N>_trials_<timestamp>.csv` — one row per trial across all input files, with derived columns added (e.g. `correct`, `response_last`, `block`, `block_switch_rep`).
- `<exp_name>_n<N>_scores_<timestamp>.csv` — one row per input file, with aggregated scores.

`<exp_name>` is read from the `exp_name` metacol of the first input. `<N>` is the count of unique participant IDs. `<timestamp>` is `YYYYMMDDhhmmss` of the run.

A log file `log_<timestamp>.log` is also written into `out/`.

## Troubleshooting

**The fastest first step is `preflight_check`** — it confirms every CSV column your config references actually exists in your data, without doing any scoring. See [`docs/CONFIGURING.md#preflight_check`](docs/CONFIGURING.md#preflight_check--validate-config--csv-before-scoring) for usage. The full required-keys-per-task cheat-sheet is in the same doc.

| Error | What it means | Fix |
| --- | --- | --- |
| `ConfigError: params is missing required key 'metacols'` | Your JSON config is missing a section, or has a typo in the section name. | Compare your JSON against the matching `*_example.json` in `tests/`. |
| `WARNING : utils : cols.X: configured CSV column 'Y' is not in this file's columns` | Your JSON says to read column `Y` for the `X` field, but `Y` isn't in the CSV. | Fix the spelling on the right-hand side of `"X": "Y"` in your JSON, or run `preflight_check` for a full list of mismatches. |
| `0/N files scored; skipped N` in the log | Every file failed to score. | Check the log for per-file error traces. Usually one of: the JSON points at column names that aren't in the CSV; the CSV is for a different experiment version. |
| `FileNotFoundError` when starting | The `out` directory's parent doesn't exist (only the leaf is auto-created). | Create the parent directory by hand, or use an `out` path that's adjacent to existing folders. |
| GUI file picker pops up unexpectedly | You forgot to pass `filelist=...`. | Pass a list of paths explicitly. |
| Output trials CSV has only metadata columns | Every `cols.*` mapping pointed at a missing column. | Look for WARNING lines in the log — each one names the misnamed column. |
| `KeyError` deep inside scoring | Your CSV is missing an expected column that the scoring code requires unconditionally (like `cols.trial`). | Check the required-keys table in [`docs/CONFIGURING.md`](docs/CONFIGURING.md#required-json-keys-per-task). |

## For contributors

If you're editing the Python source, see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for the repo layout, how the task modules share a common loop via `utils.run_task`, and how to run the regression test suite.
