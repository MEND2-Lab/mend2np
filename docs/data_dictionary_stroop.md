# Data dictionary — Stroop

The Stroop task measures interference between automatic word reading and effortful colour identification. On each trial a word is shown in one of three colours, and the participant must respond based on the *colour* (not the word's meaning). Two test types alternate within the same data file:

- **Classic** — colour-word combinations are `congruent` (e.g. the word "BLUE" rendered in blue), `incongruent` (e.g. "BLUE" rendered in pink), or `neutral` (`XXX` in some colour).
- **Emotional** — a word with affective valence is rendered in one of the three colours; condition is `negative`, `positive`, or `neutral`.

Three derived Stroop-interference contrasts are produced when the relevant condition buckets are populated.

`mend2np.stroop` writes:

- `<exp_name>_n<N>_trials_<timestamp>.csv` — one row per trial.
- `<exp_name>_n<N>_scores_<timestamp>.csv` — one row per input file.

## Shared metadata columns (both files)

| Column | Type | Description |
| --- | --- | --- |
| `id` | str | Participant ID. Source: `metacols.id`. |
| `filename` | str | Basename of the input CSV. |
| `session` | str/int | Session number. Source: `metacols.session`. |
| `datetime` | str | Date/time. Source: `metacols.datetime`. |
| `exp_name` | str | Experiment name. Source: `metacols.exp_name`. |
| `software_version` | str | PsychoPy version. Source: `metacols.software_version`. |
| `framerate` | float | Display framerate (Hz). Source: `metacols.framerate`. |
| `os` | str | Operating system. Source: `metacols.os`. |

## Trials file (`stroop_trials.csv`)

One row per trial. Same columns whether keyboard or touch — the `response` cells are normalised through `resp_mapping` to a shared option-int namespace.

### Configured columns

| Column | Type | Description |
| --- | --- | --- |
| `trial` | float | Trial counter. Source: `cols.trial`. |
| `block` | float | Block index, backfilled from PsychoPy's end-of-block markers. Source: `cols.block`. |
| `test` | str | Test type: `'classic'` or `'emotional'`. Source: `cols.test`. May be passed through an optional `test_mapping` for renaming. |
| `condition` | str | Condition within the test (see slot table below). Source: `cols.condition`. |
| `subblock` | float | Sub-block identifier. Source: `cols.subblock`. |
| `subblock_type` | str | Sub-block type label (e.g. `'Neutx2_Neg'`). Source: `cols.subblock_type`. |
| `change_color` | float | Indicator of a colour change between trials. Source: `cols.change_color`. |
| `emot_color_switch` | float | Indicator of an emotional / colour switch. Source: `cols.emot_color_switch`. |
| `this_word` | str | The word shown on the trial. Source: `cols.this_word`. |
| `this_color` | str | The colour the word was rendered in (e.g. `'hotpink'`, `'blue'`, `'saddlebrown'`). Source: `cols.this_color`. |
| `this_duration` | float | Stimulus duration (seconds). Source: `cols.this_duration`. |
| `response` | list | List of response tokens recorded on the trial. Source: `cols.response`. |
| `rt` | list[float] | List of RTs (seconds), one per response. Source: `cols.rt`. |

### Derived columns from `parse_responses`

| Column | Type | Description |
| --- | --- | --- |
| `num_responses` | float | Number of responses recorded. |
| `response_first` | float | First response, mapped to an option int (1 / 2 / 3) via `resp_mapping`. |
| `response_last` | float | Last response, mapped to an option int. Useful for inspecting self-corrections. |
| `rt_first` | float | RT of the first response (seconds). Used for scoring (standard Stroop convention). |
| `rt_last` | float | RT of the last response (seconds). |
| `correct_opt` | float | Option int the participant *should* have selected, derived from `this_color` via `color_correct_mapping` (or copied from `correct_resp` when supplied). |
| `correct` | float | 1.0 if `response_first == correct_opt`; 0.0 if they differ; NaN if either side is missing. |

## Scores file (`stroop_scores.csv`)

Score columns follow `<test>_<condition>_<metric>` plus three derived interference contrasts.

### Per-bucket metric set

For each `(test, condition)` bucket. Computed in `stroop.score_df` ([mend2np/stroop.py](../mend2np/stroop.py)).

| Metric suffix | Type | Description |
| --- | --- | --- |
| `_n_trials` | int | Trial count in the bucket. |
| `_prop_correct` | float | Mean of `correct` in the bucket. |
| `_mean_rt_correct` | float | Mean of `rt_first` restricted to correct trials. |
| `_sd_rt_correct` | float | SD of `rt_first` restricted to correct trials. |
| `_mean_rt_incorrect` | float | Mean of `rt_first` restricted to incorrect trials. |
| `_sd_rt_incorrect` | float | SD of `rt_first` restricted to incorrect trials. |

### Bucket prefix slots

| `test` slot | `condition` slots | Resulting prefixes |
| --- | --- | --- |
| `classic` | `congruent`, `incongruent`, `neutral` | `classic_congruent`, `classic_incongruent`, `classic_neutral` |
| `emotional` | `negative`, `positive`, `neutral` | `emotional_negative`, `emotional_positive`, `emotional_neutral` |

So the per-bucket columns are `classic_congruent_n_trials`, `classic_congruent_prop_correct`, `classic_congruent_mean_rt_correct`, …, `emotional_positive_sd_rt_incorrect`.

### Derived Stroop-interference contrasts

| Column | Type | Description |
| --- | --- | --- |
| `classic_stroop_interference_rt` | float | `classic_incongruent_mean_rt_correct − classic_congruent_mean_rt_correct`. The classic Stroop effect — typically ~100–200 ms in healthy adults. |
| `emotional_negative_interference_rt` | float | `emotional_negative_mean_rt_correct − emotional_neutral_mean_rt_correct`. |
| `emotional_positive_interference_rt` | float | `emotional_positive_mean_rt_correct − emotional_neutral_mean_rt_correct`. |

NaN if either operand is missing.
