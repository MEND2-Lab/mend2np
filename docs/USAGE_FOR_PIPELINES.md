# Using mend2np from a study-scale pipeline

This doc is for batch-processing many participants' data on an HPC (or any
batch environment). It covers two recipes that aren't built into the
library itself:

1. **Detect each input CSV's response modality** (keyboard vs touch) so the
   pipeline can pick the right config JSON automatically.
2. **Merge the keyboard and touch outputs** into a single trials CSV and a
   single scores CSV per task, regardless of how many runs you scored.

Both are short standalone functions — drop them into your pipeline's
utilities module and adapt as needed. They depend only on `pandas` and the
public mend2np API.

## Suggested pipeline shape

```
pipeline_run/
├── input/                          # raw CSVs, possibly hundreds, mixed tasks & modalities
│   ├── subj001_sert_keyboard.csv
│   ├── subj001_synonyms_touch.csv
│   └── ...
├── intermediate/                   # one subdir per task; mend2np writes here per-run
│   ├── sert/
│   │   ├── sert_keyboard_n12_trials_20260601...csv
│   │   ├── sert_keyboard_n12_scores_20260601...csv
│   │   ├── sert_touch_n8_trials_20260601...csv
│   │   └── sert_touch_n8_scores_20260601...csv
│   ├── synonyms/
│   └── ...
├── output/                         # merged-per-task final products
│   ├── sert_trials.csv
│   ├── sert_scores.csv
│   ├── synonyms_trials.csv
│   ├── synonyms_scores.csv
│   └── ...
```

Workflow:

1. Group input CSVs by task (by filename convention or by reading `expName`).
2. For each CSV, detect modality and dispatch to the matching config.
3. Within each task subdir, merge all the timestamped outputs into one
   trials + one scores CSV.

## Recipe 1 — detect the response modality of a CSV

PsychoPy data files emit different response columns for keyboard vs touch.
A reliable signal: which response column has populated values on trial
rows. Empty list-strings (`'[]'`) and NaN both count as "no data".

```python
import pandas as pd

# (keyboard_response_col, touch_response_col) per task. Extend this dict as
# new tasks pick up touch variants. The columns are the same ones the
# example configs reference.
_MODALITY_COLS = {
    'sert':     ('stim_response.keys',     'touch_stim_response.clicked_name'),
    'synonyms': ('key_resp_2.keys',        'word_touch_response.clicked_name'),
    'stroop':   ('trial_key_resp.keys',    'trial_mouse.clicked_name'),
    # fingosc keyboard variants use block-specific column names; for the
    # wide/test2-3 layout, the dominant-block keyboard col is the easiest tell.
    'fingosc':  ('dominant_key_resp.keys', 'mouse_Right.clicked_name'),
    # smid, pgng, bart, fept — only keyboard configs exist as of writing.
    # When touch configs are added, extend this table.
}


def detect_response_modality(csv_path:str, task:str) -> str:
    """Return 'keyboard' or 'touch' based on which response column is more populated.

    Raises KeyError if `task` isn't in `_MODALITY_COLS`. Raises ValueError if
    neither candidate column is present in the CSV.
    """
    if task not in _MODALITY_COLS:
        raise KeyError(f"no modality columns defined for task {task!r}")
    kb_col, touch_col = _MODALITY_COLS[task]

    header = pd.read_csv(csv_path, nrows=0).columns
    if kb_col not in header and touch_col not in header:
        raise ValueError(f"neither {kb_col!r} nor {touch_col!r} in {csv_path}")
    if touch_col not in header:
        return 'keyboard'
    if kb_col not in header:
        return 'touch'

    # Both columns exist — compare populated counts. PsychoPy stores "no data"
    # cells as the literal string '[]' (or NaN); anything else is a real response.
    df = pd.read_csv(csv_path, usecols=[kb_col, touch_col])

    def _populated(s):
        return s.dropna().astype(str).str.strip().ne('[]').sum()

    return 'keyboard' if _populated(df[kb_col]) >= _populated(df[touch_col]) else 'touch'
```

Wire it up to config selection:

```python
import json
from pathlib import Path
from mend2np import sert, synonyms, stroop, fingosc, bart, fept, smid, pgng

CONFIGS_DIR = Path('/path/to/mend2np/tests')   # or wherever you keep your configs

_CONFIG_FILE = {
    ('sert',     'keyboard'): 'sert_example.json',
    ('sert',     'touch'):    'sert_example_touch.json',
    ('synonyms', 'keyboard'): 'synonyms_example.json',
    ('synonyms', 'touch'):    'synonyms_example_touch.json',
    ('stroop',   'keyboard'): 'stroop_example.json',
    ('stroop',   'touch'):    'stroop_example_touch.json',
    ('fingosc',  'keyboard'): 'fingosc_example.json',
    ('fingosc',  'touch'):    'fingosc_example_touch.json',
    ('bart',     'keyboard'): 'bart_example.json',
    ('fept',     'keyboard'): 'fept_example.json',
    ('smid',     'keyboard'): 'smid_example.json',
    ('pgng',     'keyboard'): 'pgng_example.json',
}

_TASK_FN = {
    'sert': sert, 'synonyms': synonyms, 'stroop': stroop, 'fingosc': fingosc,
    'bart': bart, 'fept': fept, 'smid': smid, 'pgng': pgng,
}

def score_one(csv_path:str, task:str, out_dir:str):
    """Detect modality, load the matching config, score the file."""
    modality = (detect_response_modality(csv_path, task)
                if task in _MODALITY_COLS else 'keyboard')
    config_file = _CONFIG_FILE[(task, modality)]
    with open(CONFIGS_DIR / config_file) as f:
        params = json.load(f)
    _TASK_FN[task](params=params, filelist=[csv_path], out=out_dir)
```

A few notes:

- **Pre-flight first.** Before turning the pipeline loose on a study, run
  `mend2np.preflight_check(params, filelist, task)` for a few representative
  CSVs to catch any column-name drift in newer experiment versions.
- **Threshold tweaks.** The 50/50 split (`>= touch_col`) is a defensive
  default; if you find ambiguous CSVs that drift between modalities mid-run
  (which shouldn't happen for these tasks but might in edge cases), require
  a stronger margin (e.g. `kb_count > 2 * touch_count`).
- **For fingosc's "stacked" layout** (where both blocks share a single
  column set — see the `fingosc_example_stacked.json` config), detection
  needs a different signal: the presence of `blocks.thisN` AND the absence
  of `dominant_*` / `nondominant_*` per-block columns. Extend the helper
  with a third branch when you hit a study that uses that variant.

## Recipe 2 — merge keyboard + touch outputs into one of each

Each call to `sert(...)`, `synonyms(...)`, etc. writes timestamped CSVs to
its `out` directory: `<exp_name>_n<N>_trials_<timestamp>.csv` and
`<exp_name>_n<N>_scores_<timestamp>.csv`. After a pipeline has scored every
input file, you have a directory full of these per-batch outputs that need
to be combined.

Because the library's `format_df` standardizes column names regardless of
modality (e.g. synonyms's `response_last` is the same option-int whether
the source was a keyboard key or a touch label), the trial-level CSVs
concatenate cleanly with `pd.concat`. Likewise for scores.

```python
import glob
from pathlib import Path
import pandas as pd

def merge_task_outputs(intermediate_dir:str, out_path_trials:str, out_path_scores:str,
                       add_modality_col:bool=True):
    """Concatenate every `*_trials_*.csv` and `*_scores_*.csv` in `intermediate_dir`
    into a single combined trials CSV and a single combined scores CSV.

    If `add_modality_col=True`, the output gets a `modality` column derived from
    the source filename (presence of 'touch' substring → 'touch', else 'keyboard').
    Adjust the heuristic if your filename convention differs.
    """
    intermediate = Path(intermediate_dir)
    trial_files = sorted(intermediate.glob('*_trials_*.csv'))
    score_files = sorted(intermediate.glob('*_scores_*.csv'))

    def _read_one(path):
        df = pd.read_csv(path)
        if add_modality_col:
            df['modality'] = 'touch' if 'touch' in path.name.lower() else 'keyboard'
        return df

    trials = pd.concat([_read_one(p) for p in trial_files], ignore_index=True)
    scores = pd.concat([_read_one(p) for p in score_files], ignore_index=True)

    trials.to_csv(out_path_trials, index=False)
    scores.to_csv(out_path_scores, index=False)
    return trials, scores
```

End-to-end usage:

```python
import shutil
from pathlib import Path

PIPELINE_ROOT = Path('/scratch/me/pipeline_run')
INPUT_DIR = PIPELINE_ROOT / 'input'
INTERMEDIATE = PIPELINE_ROOT / 'intermediate'
OUTPUT = PIPELINE_ROOT / 'output'
OUTPUT.mkdir(parents=True, exist_ok=True)

# 1. Score every input CSV into its task subdir.
for csv_path in INPUT_DIR.glob('*.csv'):
    task = detect_task_from_filename(csv_path)   # whatever convention you adopt
    out_dir = INTERMEDIATE / task
    out_dir.mkdir(parents=True, exist_ok=True)
    score_one(str(csv_path), task, str(out_dir))

# 2. Merge per-task.
for task_dir in INTERMEDIATE.iterdir():
    if not task_dir.is_dir():
        continue
    task = task_dir.name
    merge_task_outputs(
        intermediate_dir=str(task_dir),
        out_path_trials=str(OUTPUT / f'{task}_trials.csv'),
        out_path_scores=str(OUTPUT / f'{task}_scores.csv'),
    )
```

Notes:

- **Column-set differences across modalities.** Sometimes a keyboard config
  passes through a column that the touch config doesn't (or vice versa).
  `pd.concat` handles this by filling NaN for the missing entries — the
  merged CSV ends up with the union of columns, which is what you want.
- **Score column unions.** A scores CSV from a sparsely-sampled run (few
  trials per condition) may have fewer condition columns than a full run
  for the same task. Same `pd.concat` behaviour applies; absent condition
  scores come out as NaN.
- **Filename order is alphabetical.** If you need participant order in the
  merged output to match a known order, sort the dataframe by `id` after
  the concat.
- **No deduplication.** If a participant was scored twice (e.g. a reprocess
  with a different config), both rows survive the merge. Add a
  `drop_duplicates(subset='id', keep='last')` step if you want only the
  most recent.

## Things mend2np itself takes care of (you don't need to)

- **Per-file logging.** Every `task()` call writes `out/log_<ts>.log` with
  per-file outcome + the summary line `scored N/M files; skipped K`. Tail
  it (or grep for `WARNING|ERROR`) to spot problems in a batch run.
- **Column-name warnings.** A configured CSV column that's missing from
  the input emits a clear warning at WARNING level — no silent dropping.
- **Output directory creation.** Pass any `out=...` path; mend2np creates
  it if it doesn't exist.
- **Filelist flexibility.** `filelist=` accepts a list of paths or a path
  to a text file with one path per line. For HPC jobs that submit one
  participant per task, just pass a 1-element list.
