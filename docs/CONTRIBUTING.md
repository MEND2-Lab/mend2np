# Contributing to mend2np

This file is for lab members editing the Python source — adding a new task module, fixing a bug, extending an existing scorer. For just using the library, see the README.

## Repo layout

```
mend2np/
├── mend2np/                    # the package source
│   ├── __init__.py             # exports: sert, pgng, bart, fept, synonyms
│   ├── utils.py                # shared helpers (logger, run_task loop, validate_params, write_out)
│   ├── sert.py
│   ├── pgng.py
│   ├── bart.py
│   ├── fept.py
│   └── synonyms.py
├── tests/
│   ├── example_data/           # sample CSVs (one per task)
│   ├── expected/               # frozen baselines for regression tests
│   ├── example_driver_*.py     # one driver per task; demonstrates usage
│   ├── *_example.json          # JSON config examples for tasks that load from JSON
│   ├── test_regression.py      # end-to-end regression suite
│   └── out/                    # output dir created at run time (gitignored)
├── docs/
│   ├── CONFIGURING.md          # JSON config guide for non-programmers
│   └── CONTRIBUTING.md         # this file
├── README.md
└── pyproject.toml
```

## How the task modules are shaped

Each task module follows the same skeleton:

```python
REQUIRED_PARAMS = { ... }  # schema for validate_params

def task_name(params, out, write, filelist, formatted, log, ...):
    setup_logger(name='root', out=out, level=log).info('start')
    validate_params(params, REQUIRED_PARAMS)

    def process_one(filepath, params, logger):
        df = pd.read_csv(filepath)
        if not formatted:
            df = format_df(df, params)
        # task-specific transforms
        scores_row = pd.concat([get_meta_cols(df, params), score_df(df)], axis=1)
        return df, scores_row   # or (None, scores_row) if no per-trial output

    return run_task(
        params=params, filelist=filelist, out=out, write=write, log=log,
        process_file_fn=process_one,
    )
```

`utils.run_task` owns the boilerplate that every module shares: resolving the `filelist` argument, the per-file try/except, accumulating combined trial- and score-level dataframes, and calling `write_out` at the end. Each task's `process_one` callable contains only the task-specific logic.

## Adding a new task

1. Create `mend2np/<task>.py`. Mirror the structure of `bart.py` (the simplest existing module) — define `REQUIRED_PARAMS`, the entry function calling `run_task`, plus the task-specific `format_df` and `score_df` helpers.
2. Add the function to `mend2np/__init__.py`'s exports.
3. Add an example CSV to `tests/example_data/` and a driver script to `tests/`. The driver must pass `filelist=[...]` explicitly so testing isn't blocked by a GUI file picker.
4. Add an entry to `DRIVERS` in `tests/test_regression.py` and snapshot the current output into `tests/expected/<task>_scores.csv` (and `_trials.csv` if applicable).
5. Run `python tests/test_regression.py` — your new task should appear in the pass list.

## Running tests

```bash
python tests/test_regression.py
```

The regression test runs each example driver, locates its timestamped output via glob, and compares against `tests/expected/<task>_*.csv` using `pd.testing.assert_frame_equal` (with `rtol=atol=1e-9`, ignoring dtype).

**When you intentionally change scoring behavior** (e.g. fixing a bug whose fix legitimately alters output values), re-snapshot the baseline:

```bash
rm tests/out/*.csv
python tests/example_driver_<task>.py
cp tests/out/<output_pattern>.csv tests/expected/<task>_scores.csv
# (and trials.csv if applicable)
```

Then describe the diff in the commit message so future-you knows why the baseline moved.

## Conventions

- **No mutation of input dataframes.** `process_one` callables receive a fresh `df = pd.read_csv(...)`; subsequent transforms should `df = transform(df)` rather than `df.method(inplace=True)`.
- **Row-by-row iteration is OK when the state is sequential** (the PGNG response classifiers genuinely need prior-row state). Otherwise prefer `.apply` on whole Series, or boolean-mask assignments. Never use `df.at[i, col] = value` inside `iterrows()` — it triggers SettingWithCopyWarning whenever the caller passes a slice.
- **Module-level `logger = logging.getLogger('root')`** so helper functions in the module can use `logger.warning(...)` without a `global` declaration. `setup_logger` configures the same root logger by name.
- **Schemas only need to cover the top-level structural keys** (`metacols`, `cols`, `blocks`). Per-key validation deeper than that is more trouble than it's worth — the column-existence checks in `format_df` cover the rest gracefully.
- **`_`-prefixed keys in JSON configs are skipped.** Use `"_comment": "..."` to annotate sections without breaking the scoring code.
