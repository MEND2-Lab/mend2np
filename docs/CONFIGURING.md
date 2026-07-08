# Configuring a JSON config for a new experiment version

This guide is for those who need to score a CSV from a new (or updated) experiment but don't write Python. You'll edit a JSON file and run a driver script — nothing else.

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
- `bart_example.json` for BART
- `fept_example.json` for FEPT
- `synonyms_example.json` for synonyms
- `fingosc_example.json` for Finger Oscillation (keyboard, wide-format); also `fingosc_example_touch.json` (mouse/touch) and `fingosc_example_stacked.json` (single-column-set layout)
- `smid_example.json` for SMID
- `stroop_example.json` for Stroop (keyboard); also `stroop_example_touch.json`

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

## Required JSON keys per task

This is the cheat-sheet for "what do I have to include in my JSON for X output column to appear in my output?". The left-hand-side names below are the standard names mend2np uses internally; you fill in the right-hand sides with your CSV's matching column names.

In the tables: **REQ** = the scoring code fails or silently produces nothing without it. **OPT** = passed through to the trials output (and used for some bonus score columns) if present, ignored if absent. Anything not listed is computed by mend2np itself and doesn't need a config entry.

### sert ([example](../tests/sert_example.json))

| Key | Required? | Drives... |
| --- | --- | --- |
| `metacols.id` | OPT but recommended | Participant ID; also the basis of `nN` in the output filename |
| `metacols.session`, `datetime`, `exp_name`, `software_version`, `framerate`, `os` | OPT | Carried into trials + scores output |
| `cols.trial` | **REQ** | Used to mask off non-trial rows; without it `format_df` raises `KeyError`. Also derives `block` (`trial // 10 + 1`) when `cols.block` is absent |
| `cols.response` | **REQ for scores** | Source of `first_response`, `last_response`, `correct` |
| `cols.rt` | **REQ for scores** | Source of `first_response_rt`, `last_response_rt`, `correct_resp_rt` |
| `cols.correct_resp` | **REQ for scores** | Source of `correct` flag and `correct_resp_index` |
| `cols.event_type` | **REQ for scores** | Primary `groupby` key in `score_df` — missing it leaves the scores file with only metadata |
| `cols.cue` | **REQ for scores** | Drives trial- and block-level switch/repeat and per-cue scores |
| `cols.block` | OPT | A source-provided block index; used instead of `trial // 10` when present (e.g. when the trial counter resets per block) |
| `cols.left_choice`, `middle_choice`, `right_choice` | OPT | Each gets expanded into `_class`/`_type`/`_color`/`_shape` derived columns |
| `cols.stim_onset`, `stim_offset`, `iti_onset`, `iti_offset`, `iti_dur`, `cue_dur`, `stim_with_cue_dur` | OPT | Passed through to trials output for timing analysis |

Trial-level switch/repeat scores (`<event_type>_trial_*`) are always produced. Block-level switch/repeat scores are opt-in via the `block_switch_rep=True` argument to `sert()` — leave it off for sources without fixed switch/repeat blocks (e.g. MetricWire).

### pgng ([example](../tests/pgng_example.json))

PGNG nests its configuration under `params['blocks']` — every block must be configured independently.

| Key | Required? | Drives... |
| --- | --- | --- |
| `metacols.id` | OPT but recommended | Participant ID + filename `nN` |
| `metacols.session`, `datetime`, `exp_name`, `software_version`, `framerate`, `os` | OPT | Carried into trials + scores |
| `blocks.<B>.cols.trial` | **REQ** (psychopy) | Per-block mask of non-trial rows |
| `blocks.<B>.cols.block` | **REQ** (eprime only) | Per-block mask replacement on E-Prime data |
| `blocks.<B>.cols.stimuli` | **REQ** | Used to classify trials as target / lure / non-target |
| `blocks.<B>.cols.response` | **REQ** | Used to classify responses (hit / miss / commission / etc.) |
| `blocks.<B>.cols.rt` | **REQ** | Source of `rt_adj` and all RT-based scores |
| `blocks.<B>.cols.stop_time` | **REQ for `gs` blocks** | Override of `stim_dur` for Go/Stop trials |
| `blocks.<B>.metavars.stim_targ_names` | **REQ** | List of target stimulus filenames |
| `blocks.<B>.metavars.resp_key` | **REQ** | The keyboard key (or list of keys) participants press for a "go" |
| `blocks.<B>.metavars.type` | **REQ** | One of `'go'`, `'gng'`, `'gs'` — picks the classifier |
| `blocks.<B>.metavars.stim_dur` | OPT but recommended | Static stimulus duration; needed when CSV lacks a `stim_start` column |
| `blocks.<B>.metavars.stop` | **REQ for `gs` blocks** | Filename of the stop stimulus |

### bart ([example](../tests/bart_example.json))

| Key | Required? | Drives... |
| --- | --- | --- |
| `metacols.id` | OPT but recommended | Participant ID + filename `nN` |
| `metacols.session`, `datetime`, `exp_name`, `software_version`, `framerate`, `os` | OPT | Carried into output |
| `cols.trial` | **REQ** | Mask of non-trial rows |
| `cols.nPumps` | **REQ for scores** | Per-trial pump count |
| `cols.popped` | **REQ for scores** | Boolean — did the balloon explode? |
| `cols.earnings` | **REQ for scores** | Per-trial money won |
| `cols.rt` | **REQ for RT scores** | List of click-time series, converted to pump-latency deltas in `format_df` |

### fept ([example](../tests/fept_example.json))

| Key | Required? | Drives... |
| --- | --- | --- |
| `metacols.id` | OPT but recommended | Participant ID + filename `nN` |
| `metacols.*` | OPT | Carried into output |
| `blocks.<B>.cols.stimuli` | **REQ** | Per-block mask AND source for stim_class parsing |
| `blocks.<B>.cols.response` | **REQ for scores** | Used to compute correct/incorrect counts |
| `blocks.<B>.cols.correct_response` | **REQ for scores** | Used to compute correct/incorrect counts |
| `blocks.<B>.cols.rt` | **REQ for RT scores** | Source of `rt_mean`, `rt_sd`, etc. |
| `blocks.<B>.metavars.type` | **REQ for scores** | Becomes the column prefix in scores (e.g. `faces_correct_ct`) |
| `blocks.<B>.metavars.stimulus_duration`, `mask_duration` | OPT | Used to compute `rt_global` |
| `blocks.<B>.key_labels` | OPT | Maps raw keypress codes (`'k'`, `'space'`) to readable labels (`'happy'`, `'sad'`) for misclassification reporting |
| `blocks.<B>.stim_class_map` | OPT | Per-category regex map (emotion / race / sex / etc.) — without it, per-category scores are skipped |

### stroop ([example](../tests/stroop_example.json))

Stroop alternates classic and emotional sub-blocks within one data stream — the `test` column on each row tells you which kind of trial it was. The config is flat (one set of `cols`, not per-block). Both keyboard and touch CSVs are supported by swapping `cols.response` / `cols.rt` and using the same `resp_mapping` (which standardizes keyboard keys and touch shape names to a shared 1/2/3 option-int namespace, same approach as synonyms). Correctness is derived from the stimulus colour via `color_correct_mapping` — the bundled CSVs don't ship an explicit correct-response column for real trials.

| Key | Required? | Drives... |
| --- | --- | --- |
| `metacols.id` | OPT but recommended | Participant ID + filename `nN` |
| `metacols.*` | OPT | Standard metadata carried into output |
| `cols.trial` | **REQ** | Mask of trial rows |
| `cols.test` | **REQ for scores** | Block-type axis (`classic` vs `emotional`) in the score grid |
| `cols.condition` | **REQ for scores** | Condition axis (`congruent`/`incongruent`/`neutral` for classic; `negative`/`positive`/`neutral` for emotional) |
| `cols.this_color` | **REQ for correctness** | Combined with `color_correct_mapping` to derive `correct_opt`. Without this, `correct` / `prop_correct` / RT-by-correctness scores are all NaN. |
| `cols.this_word` | OPT | Carried into trials output |
| `cols.this_duration`, `block`, `subblock`, `subblock_type`, `change_color`, `emot_color_switch` | OPT | Per-trial condition identifiers carried into trials output for downstream analysis |
| `cols.response` | **REQ for scores** | Source of `response_first`/`last`; map through `resp_mapping` to derive the option int |
| `cols.rt` | **REQ for RT scores** | Source of `rt_first`/`last`. Stroop convention: scoring uses `rt_first` (the initial reaction). |
| `cols.correct_resp` | OPT | If your CSV has an explicit correct-response column, set this and the colour-mapping path is skipped. |
| `resp_mapping` | OPT (recommended) | Top-level dict mapping response labels → option ints. Without it, a default mapping covering `j`/`k`/`l` and `trial_opt{1,2,3}_shape` / `prac_opt{1,2,3}_shape` is used (and a warning logged). |
| `color_correct_mapping` | **REQ for correctness** | Top-level dict mapping `this_color` values → the correct option int. Example: `{"blue": 1, "saddlebrown": 2, "hotpink": 3}`. |

The score column names follow `<test>_<condition>_<metric>` — for example `classic_incongruent_mean_rt_correct`, `emotional_negative_prop_correct`. Three derived columns capture standard Stroop-interference contrasts (on mean correct-RT):

- `classic_stroop_interference_rt` = `classic_incongruent_mean_rt_correct − classic_congruent_mean_rt_correct` (the classic Stroop effect — typically ~100-200 ms for healthy adults)
- `emotional_negative_interference_rt` = `emotional_negative_mean_rt_correct − emotional_neutral_mean_rt_correct`
- `emotional_positive_interference_rt` = `emotional_positive_mean_rt_correct − emotional_neutral_mean_rt_correct`

The trial-level output preserves every per-trial condition identifier you listed (`this_word`, `this_color`, `this_duration`, `test`, `subblock`, `subblock_type`, `condition`, `change_color`, `emot_color_switch`) so you can compute additional breakdowns (e.g. by `change_color` or `emot_color_switch`) directly with pandas without re-running the scorer.

### smid ([example](../tests/smid_example.json))

SMID has two blocks — practice (`p`-prefixed PsychoPy columns) and real (unprefixed). Both share the same `metacols` (including the participant's selected `charity_name` and `staff_name`, which are carried through to both outputs). Per-block columns name the prime image, benefactor, probe key/RT, etc.

| Key | Required? | Drives... |
| --- | --- | --- |
| `metacols.id` | OPT but recommended | Participant ID + filename `nN` |
| `metacols.charity_name` | OPT but recommended | Carried into trials + scores so each row knows who the participant was working for |
| `metacols.staff_name` | OPT but recommended | Same — the charity representative the participant chose |
| `metacols.*` | OPT | Standard metadata carried into output |
| `blocks.<B>.cols.trial` | **REQ** | Per-block mask of trial rows |
| `blocks.<B>.cols.prime` | **REQ for scores** | Image filename — decoded into `reward_type` (gain/lose/neither) + `amount` (0/0.2/5) |
| `blocks.<B>.cols.benefactor` | **REQ for scores** | `'YOURSELF'` → non-social (self), `'NAME'` → social (charity) |
| `blocks.<B>.cols.probe_rt` | **REQ for RT scores** | Source of `mean_rt`, `sd_rt`. Non-null RT also serves as the fallback "correct" signal when `probe_response` is absent. |
| `blocks.<B>.cols.probe_response` | OPT but preferred | Explicit boolean for correctness on real trials. Auto-fallback chain: this → `probe_rt.notna()` → `feedback_correct.notna()`. |
| `blocks.<B>.cols.feedback_correct`, `feedback_incorrect`, `feedback_spoiled` | OPT | Passed through to trials output; also used as a last-resort "correct" fallback for older CSV variants that don't record probe_response or probe_rt. |
| `blocks.<B>.cols.probe_key`, `pre_key`, `post_probe_key` | OPT | Carried into trials output for downstream inspection |
| `blocks.<B>.cols.probe_duration`, `self_earnings`, `charity_earnings` | OPT | Per-trial bookkeeping passed through to trials output |
| `blocks.<B>.metavars.phase` | OPT (recommended) | Static label broadcast onto every row of the block; useful for filtering downstream |

The score column names follow the pattern `<social>_<type>[_<amount>]_<metric>`. For example: `self_gain_big_mean_rt`, `charity_lose_small_n_probes`, `self_neither_prop_correct` (`_<amount>` is suppressed for `neither` trials since they always have amount=0). The block name is not used as a prefix since practice trials are filtered out before scoring, leaving only the single "real" block.

**Practice trials are excluded from the scores output** (rows where `metavars.phase == 'practice'` are dropped before scoring). They still appear in the trials-level output, just not in the aggregate scores. If you want practice trials scored too, set their block's `metavars.phase` to something other than `"practice"`.

### fingosc ([example](../tests/fingosc_example.json))

Finger Oscillation has two blocks (dominant hand, then non-dominant). The CSV column layout varies a lot across experiment versions, so two distinct shapes are supported via the per-block config:

- **Wide** — each block has its own column names (e.g. `dominant_key_resp.rt` vs `nondominant_key_resp.rt`). Row mask is "rows where this block's `trial` column is non-null", the same pattern as pgng. Examples: [`fingosc_example.json`](../tests/fingosc_example.json) (keyboard), [`fingosc_example_touch.json`](../tests/fingosc_example_touch.json) (mouse/touch).
- **Stacked** — both blocks share the same column names and are stacked vertically in the CSV. Each block declares its position via `metavars.block_index` (`0` or `1`) and shares a common `metavars.block_marker_col` (a column whose non-null cells mark the END of each block). Row mask is computed by cumulative-counting marker rows. Example: [`fingosc_example_stacked.json`](../tests/fingosc_example_stacked.json).

| Key | Required? | Drives... |
| --- | --- | --- |
| `metacols.id` | OPT but recommended | Participant ID + filename `nN` |
| `metacols.*` | OPT | Carried into output |
| `blocks.<B>.cols.trial` | **REQ** (wide mode) | Per-block mask of trial rows |
| `blocks.<B>.cols.response` | OPT | Carried into trials output |
| `blocks.<B>.cols.rt` | **REQ for scores** | Source of `<B>_mean_rt`, `<B>_sd_rt` |
| `blocks.<B>.metavars.hand` | OPT (recommended) | Broadcast onto every row of the trials output. Useful for joining the per-block labels back to a hand identity. |
| `blocks.<B>.metavars.block_index` | **REQ** (stacked mode) | Which block (0 or 1) these rows belong to in a CSV where both blocks share columns |
| `blocks.<B>.metavars.block_marker_col` | **REQ** (stacked mode) | CSV column whose non-null cells mark each block's END (e.g. `blocks.thisN`) |

The block key itself (the `<B>` above — e.g. `"dominant"`, `"nondominant"`) is used as the column prefix in the scores output, so naming your blocks `"dominant"` and `"nondominant"` gives columns like `dominant_mean_rt`, `nondominant_mean_rt`, etc.

**Caveat on hand-dominance labelling.** The block labels you set (`hand: 'dominant'` etc.) are trusted as-is — the scorer does no automatic verification. If the participant instructions were inconsistent across sessions, you'll want to compare per-block RTs to a known external dominance measure before relying on the labels.

### synonyms ([example](../tests/synonyms_example.json))

| Key | Required? | Drives... |
| --- | --- | --- |
| `metacols.id` | OPT but recommended | Participant ID + filename `nN` |
| `metacols.*` | OPT | Carried into output |
| `cols.trial` | **REQ** | Mask of non-trial rows |
| `cols.response` | **REQ for scores** | Source of `response_last`, `correct` |
| `cols.rt` | **REQ for scores** | Source of `rt_last` |
| `cols.correct_resp` | **REQ for scores** | Determines `correct` flag |
| `cols.stimuli` | OPT | Carried into trials output |
| `cols.stim_dur` | OPT | Carried into trials output |
| `resp_mapping` | OPT | Top-level dict mapping `{key_or_label: option_index}`. If absent, a default mapping is used and a warning is logged. |

## Debugging tools

### `preflight_check` — validate config + CSV before scoring

Before running a full scoring pass, run this from Python to confirm every column referenced in your JSON actually exists in your CSV:

```python
import json
from mend2np import preflight_check

with open('my_configs/sert_v9.json') as f:
    params = json.load(f)

preflight_check(
    params=params,
    filelist=['data/subject1.csv', 'data/subject2.csv'],
    task='sert',
)
```

The output looks like:

```
preflight_check (sert) — 23 configured column references

[OK]    data/subject1.csv
[ISSUES] data/subject2.csv
        cols.response  ->  'stim_response.keys' (not in CSV)
        cols.cue       ->  'cue' (not in CSV)
```

`[OK]` files are ready to score; `[ISSUES]` lines name every config path whose CSV column wasn't found.

### Runtime warnings

During scoring, mend2np now emits a WARNING log line every time a config-mapped CSV column isn't found in the input. Example:

```
WARNING : utils : cols.response: configured CSV column 'stim_response.keys' is not in
this file's columns — 'response' will be missing from the output.
Check the spelling in your config, or set this entry's value to '' to skip silently.
```

These appear in both stdout and `out/log_<timestamp>.log`. If you intentionally want to disable a config entry rather than fix it, set its value to an empty string and the warning won't fire.

### Reading the log file

Every run writes `out/log_<timestamp>.log`. Useful patterns to grep for:

- `WARNING` — every time a configured column was missing, or a fallback was used.
- `ERROR` — a single file failed to score. The next lines show the traceback.
- `scored N/M files; skipped K` — the final summary line. If N==0 every file failed; check WARNINGs and ERRORs above.
