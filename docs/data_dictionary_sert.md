# Data dictionary — SERT

The Suicide Emotion Rigidity Task (SERT) measures cognitive flexibility under emotional content. On each trial three image options are shown plus a *cue* indicating which dimension to attend to (e.g. shape, colour, or lethality). Switch-cost contrasts (switch − repeat differences in RT and accuracy) are the primary outputs, and are computed at two grains:

- **Trial-level** (always produced, `<event_type>_trial_*` columns): a trial is *switch* if its cue differs from the immediately preceding trial's cue within its block, *repeat* if it matches, and *first* for the first trial of a block (no predecessor; excluded from the contrast). Captures the local, trial-to-trial cost of reconfiguring to a new rule.
- **Block-level** (opt-in via `block_switch_rep=True`, `<event_type>_switch/repeat/switch_cost_*` columns): a whole block is *switch* if the cue changes anywhere within it, *repeat* if every trial shares one cue. Suits designs with fixed switch/repeat blocks (e.g. the Pavlovia SERT's 10-trial blocks); it captures the tonic cost of a mixed-cue context. Off by default because some sources (e.g. MetricWire) don't run fixed blocks, which would collapse every block to *switch*.

Blocks are taken from a configured `cols.block` column when present, otherwise derived as fixed 10-trial runs (`trial // 10 + 1`).

`mend2np.sert` writes:

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

## Trials file (`sert_trials.csv`)

One row per trial.

### Configured columns

| Column | Type | Description |
| --- | --- | --- |
| `event_type` | str | Experiment-defined trial category (top-level grouping in scoring). Source: `cols.event_type`. |
| `cue` | str | Cue dimension this trial used (e.g. `'color'`, `'shape'`, `'lethality'`). Source: `cols.cue`. The value `'lethal'` is normalised to `'lethality'`. |
| `correct_resp` | float | Index (1-based) of the option that was correct. Source: `cols.correct_resp`. |
| `response` | list | List of response tokens recorded on the trial. For touch CSVs, raw labels (`'LeftImage'`/`'MiddleImage'`/`'RightImage'`) are mapped to `1/2/3`. Source: `cols.response`. |
| `rt` | list[float] | List of RTs (seconds), one per response. Source: `cols.rt`. |
| `trial` | float | Trial counter. Used to derive `block` when no `cols.block` is configured (`trial // 10 + 1`). Source: `cols.trial`. |
| `block` | float | Optional. A block index supplied by the source (e.g. when a per-block trial counter resets). Source: `cols.block`. When mapped, it is used as-is instead of deriving blocks from `trial`. |
| `left_choice` | str | Left option's stimulus filename. Source: `cols.left_choice`. |
| `middle_choice` | str | Middle option's stimulus filename. Source: `cols.middle_choice`. |
| `right_choice` | str | Right option's stimulus filename. Source: `cols.right_choice`. |
| `stim_onset` | float | Stimulus onset time. Source: `cols.stim_onset`. |
| `stim_offset` | float | Stimulus offset time. Source: `cols.stim_offset`. |
| `iti_onset` | float | ITI onset time. Source: `cols.iti_onset`. |
| `iti_offset` | float | ITI offset time. Source: `cols.iti_offset`. |
| `iti_dur` | float | ITI duration (seconds). Source: `cols.iti_dur`. |
| `cue_dur` | float | Cue duration (seconds). Source: `cols.cue_dur`. |
| `stim_with_cue_dur` | float | Combined stimulus + cue duration (seconds). Source: `cols.stim_with_cue_dur`. |

### Derived columns from `parse_choice_columns`

For each side (`left`, `middle`, `right`), the stimulus filename is parsed into 4 components. Filenames look like `Lethal_Pills_Blue_Oval.png`.

| Column | Type | Description |
| --- | --- | --- |
| `<side>_choice_class` | str | Object class — one of `'safe'`, `'inert'`, `'lethal'`, or NaN. |
| `<side>_choice_type` | str | Type tokens (everything between class and color/shape, joined by `_`). |
| `<side>_choice_color` | str | Object colour — one of `'orange'`, `'blue'`, or NaN. |
| `<side>_choice_shape` | str | Object shape — one of `'oval'`, `'rhom'`, `'rect'`, or NaN (`'rectangle'` is normalised to `'rect'`). |

### Derived columns from `add_blocks` and `add_trial_switch_rep`

| Column | Type | Description |
| --- | --- | --- |
| `block` | float | Block index. Uses `cols.block` when configured, else `(trial // 10) + 1`. |
| `trial_switch_rep` | str | Trial-level label: `'switch'` if this trial's `cue` differs from the previous trial's cue within the same block, `'repeat'` if it matches, `'first'` for the first trial of a block. Always added. |

### Derived columns from `add_block_switch_rep` (only when `block_switch_rep=True`)

| Column | Type | Description |
| --- | --- | --- |
| `block_nunique_cues` | int | Number of distinct `cue` values within this row's block. |
| `block_switch_rep` | str | Block-level label: `'switch'` if `block_nunique_cues > 1`, else `'repeat'`. |

### Derived columns from `parse_responses`

| Column | Type | Description |
| --- | --- | --- |
| `num_responses` | float | Number of responses recorded on the trial. |
| `first_response` | float | First response (option int). |
| `first_response_rt` | float | RT of the first response (seconds). |
| `last_response` | float | Last response (option int). |
| `last_response_rt` | float | RT of the last response (seconds). |
| `correct` | float | 1.0 if `correct_resp` appears anywhere in the response list, else 0.0. |
| `correct_resp_index` | float | Index within the response list at which `correct_resp` first appeared (NaN if never). |
| `correct_resp_rt` | float | RT corresponding to that index. |

## Scores file (`sert_scores.csv`)

Score columns follow the pattern `<event_type>[_<level>][_<switch_rep>][_<cue>]_<metric>` plus parallel `switch_cost` difference columns. Computed in `sert.score_df` ([mend2np/sert.py](../mend2np/sert.py)).

The trial-level grid (`<level>` = `trial`) is always emitted. The block-level grid (`<level>` empty) is emitted only when the file was scored with `block_switch_rep=True`; otherwise those columns are absent.

### Per-bucket metric set

| Metric suffix | Type | Description |
| --- | --- | --- |
| `_num_trials` | int | Number of trials in this bucket. |
| `_num_correct` | int | Number correct (`correct == 1`). |
| `_accuracy` | float | `num_correct / num_trials`. |
| `_mean_first_rt` | float | Mean of `first_response_rt`. |
| `_median_first_rt` | float | Median of `first_response_rt`. |
| `_std_first_rt` | float | SD of `first_response_rt`. |
| `_mean_correct_resp_rt` | float | Mean of `correct_resp_rt`, restricted to correct trials. |
| `_median_correct_resp_rt` | float | Median of `correct_resp_rt`, restricted to correct trials. |
| `_std_correct_resp_rt` | float | SD of `correct_resp_rt`, restricted to correct trials. |

### Bucket prefix patterns

Each prefix combines with the metric suffixes above to form a column name (e.g. `objects_trial_repeat_color_accuracy`).

| Prefix pattern | Bucket = trials where… |
| --- | --- |
| `<event_type>` | matches this `event_type` (all trials of that type). |
| `<event_type>_trial_<switch_rep>` | matches `event_type` AND `trial_switch_rep`. `<switch_rep>` ∈ {`switch`, `repeat`}. Always emitted. |
| `<event_type>_trial_<switch_rep>_<cue>` | matches `event_type` AND `trial_switch_rep` AND `cue`. Always emitted. |
| `<event_type>_<switch_rep>` | matches `event_type` AND `block_switch_rep`. Only when `block_switch_rep=True`. |
| `<event_type>_<switch_rep>_<cue>` | matches `event_type` AND `block_switch_rep` AND `cue`. Only when `block_switch_rep=True`. |

`<event_type>` is whatever distinct values your data carry. The bundled example data uses only `objects`. `<cue>` is whatever distinct values appear in the `cue` column — typically `color`, `shape`, `lethality` (after the `lethal` → `lethality` normalisation). Per-cue buckets are written for every distinct cue present in the data. The trial-level `first` label is not scored as its own bucket.

### Switch-cost contrast columns

For each `<event_type>`, the scorer computes `switch − repeat` differences for a fixed set of (cue, metric) combinations. These exist for the trial-level grid (always) and, when `block_switch_rep=True`, the block-level grid.

| Pattern | Meaning |
| --- | --- |
| `<event_type>_trial_switch_cost_<metric>` | `<event_type>_trial_switch_<metric>` − `<event_type>_trial_repeat_<metric>` (trial-level, no cue split). |
| `<event_type>_trial_switch_cost_<cue>_<metric>` | per-cue trial-level switch cost. |
| `<event_type>_switch_cost_<metric>` | `<event_type>_switch_<metric>` − `<event_type>_repeat_<metric>` (block-level, no cue split; only when `block_switch_rep=True`). |
| `<event_type>_switch_cost_<cue>_<metric>` | per-cue block-level switch cost (only when `block_switch_rep=True`). |

`<cue>` is hardcoded to `{color, shape, lethality}` for switch-cost contrasts (see `_SWITCH_COST_CUES` in [mend2np/sert.py](../mend2np/sert.py)). `<metric>` is hardcoded to `{mean_first_rt, median_first_rt, mean_correct_resp_rt, median_correct_resp_rt, accuracy, num_correct}` (`_SWITCH_COST_METRICS`). Switch-cost columns where either operand is missing produce `NaN` — common at the per-cue trial level for short runs that lack a switch (or repeat) trial of a given cue.
