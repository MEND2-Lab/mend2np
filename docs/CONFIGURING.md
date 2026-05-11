# Configuring a JSON config for a new experiment version

This guide is for lab members who need to score a CSV from a new (or updated) experiment but don't write Python. You'll edit a JSON file and run a driver script — nothing else.

The JSON config does one job: it tells mend2np what your experiment's CSV column names mean. PsychoPy and E-Prime versions name their output columns slightly differently across experiment versions, and that's the whole reason this file exists.

## What you'll need

- The CSV file you want to score.
- A spreadsheet program (Excel, Numbers, Google Sheets) to inspect the CSV.
- A text editor (TextEdit, Notepad, VS Code) to edit the JSON.

## Step 1 — open the CSV and look at the column names

Open your CSV in a spreadsheet. The first row is the header — those are your column names. You don't need to read the data; just look at the column names so you know which ones exist.

Write down (or just notice) the columns that look like:

- The participant ID column (often `participant`, `subjectID`, etc.).
- The session column (often `session`).
- The date/time column (often `date`, `datetime`).
- The experiment name column (often `expName`).
- The PsychoPy version / framerate / OS columns (these are usually called exactly `psychopyVersion`, `frameRate`, `OS`).
- The trial-counter column (one that has a sequential number per real trial). For PGNG this is per-block, e.g. `PGNGS_B1.thisTrialN`.
- The response-key column (often `key_resp.keys` or `<task>_resp.keys`).
- The response-time column (the one ending in `.rt`).
- The correct-answer column (often `correct`, `correct_resp`, `Correct_Response`).
- The stimulus column.

## Step 2 — copy the matching example JSON

Find the example file in `tests/` that matches your task:

- `sert_example.json` for SERT (keyboard responses)
- `sert_example_touch.json` for SERT (touchscreen)
- `pgng_example.json` for PGNG
- `synonyms_example.json` for synonyms
- (BART and FEPT configs live in their Python driver scripts at `tests/example_driver_bart.py` / `_fept.py` — copy the `params = { ... }` block from there.)

Save your copy somewhere outside `tests/` (e.g. `my_configs/sert_v9.json`) so you don't accidentally modify the example.

## Step 3 — edit your copy

Open `my_configs/sert_v9.json` and look at it. You'll see something like:

```json
{
  "metacols": {
    "id": "participant",
    "session": "session",
    "datetime": "date",
    ...
  },
  "cols": {
    "correct_resp": "correct_resp",
    "response": "stim_response.keys",
    "rt": "stim_response.rt",
    ...
  }
}
```

The rule is simple: **the left side is what mend2np will call the column internally; the right side is what your CSV calls it.** Look at the right-hand side and check that each value really exists as a column header in your CSV. When a name is wrong, edit only the right-hand side.

Examples of edits:
- If your CSV has `subjectID` instead of `participant`: change `"id": "participant"` to `"id": "subjectID"`.
- If your CSV has `RT` (uppercase) instead of `stim_response.rt`: change `"rt": "stim_response.rt"` to `"rt": "RT"`.

Don't change the left-hand side names — those are what the scoring code looks up internally.

### Adding annotations (optional)

JSON doesn't support inline comments, but you can add helper notes by inserting keys that start with `_` (e.g. `"_comment": "this section maps metadata columns"`). mend2np ignores any key whose name starts with an underscore, so they're safe to leave in the file.

See [`tests/sert_example_ANNOTATED.json`](../tests/sert_example_ANNOTATED.json) for a fully annotated example.

## Step 4 — point a driver script at your CSV and config

Copy one of the example driver scripts, e.g. `tests/example_driver_sert.py`, to your own folder. The interesting lines are:

```python
filelist = [
    str(data_dir / 'example_data_psychopy_sert_1.csv'),
    ...
]
with open(HERE / 'sert_example.json', 'r') as f:
    params = json.load(f)
sert(params=params, formatted=False, out=str(out_dir), filelist=filelist)
```

Change those to point at your CSV(s), your config, and your desired output directory. Then run `python my_driver.py`.

## Step 5 — check the output

Look in your output directory. You should see:

- `<exp_name>_n<N>_trials_<timestamp>.csv` — one row per trial, including derived columns added by mend2np (like `correct`, `response_last`, `block_switch_rep`).
- `<exp_name>_n<N>_scores_<timestamp>.csv` — one row per participant with summary scores.
- `log_<timestamp>.log` — informational + any error messages.

If the trials CSV has only metadata columns (no `response`, `rt`, etc.), the config's column names didn't match the CSV. Open the log file and look at the last few lines for the specific column it couldn't find.

## Common config mistakes

- **Typo in a left-hand-side name.** e.g. `"metcols"` instead of `"metacols"` at the top level. The library now gives a clear `ConfigError` for these.
- **Right-hand-side name doesn't match the CSV exactly.** Column names are case-sensitive and dot-sensitive (`stim_response.rt` ≠ `stim_response.RT` ≠ `stim_response_rt`).
- **Trailing comma in JSON.** JSON, unlike Python, does not allow a comma after the last item in a list or object. If your editor highlights an error, this is usually it.
- **Pointing at the touch config when the data is keyboard (or vice versa).** SERT has separate configs for the two — pick the one matching how the participant responded.
