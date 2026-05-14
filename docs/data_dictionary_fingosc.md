# Data dictionary — Finger Oscillation (fingosc)

The Finger Oscillation task measures tapping speed. Two blocks are run: one with the *dominant* hand and one with the *non-dominant* hand. Within each block, the participant taps as fast as they can — by pressing a key or by clicking a button on a touchscreen — for a fixed number of taps. One CSV row is emitted per tap.

`mend2np.fingosc` writes:

- `<exp_name>_n<N>_trials_<timestamp>.csv` — one row per tap across all input files.
- `<exp_name>_n<N>_scores_<timestamp>.csv` — one row per input file with per-block tap counts and RT summaries.

Three CSV shapes are supported (keyboard wide, touch wide, stacked) and produce identical output columns; only the per-block config differs. See [`CONFIGURING.md#fingosc`](CONFIGURING.md#fingosc-example) for details.

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

## Trials file (`fingosc_trials.csv`)

One row per tap.

| Column | Type | Description |
| --- | --- | --- |
| (shared metadata) | — | See block above. |
| `response` | str | The response key or shape name recorded for this tap. Source: `blocks.<B>.cols.response`. |
| `rt` | float | Time between this tap and the previous tap (seconds). Source: `blocks.<B>.cols.rt`. Touch variants may store this as a list-string in the raw CSV; the scorer parses it back to a scalar float. |
| `trial` | float | Tap counter (0-indexed). Source: `blocks.<B>.cols.trial`. |
| `hand` | str | Hand label broadcast from the config: typically `'dominant'` or `'nondominant'`. Source: `blocks.<B>.metavars.hand`. (Trusted as-is — see the caveat in [CONFIGURING.md](CONFIGURING.md#fingosc-example) about hand-dominance labelling.) |
| `block` | str | The block key from the config (the JSON object key under `blocks`). For the bundled configs, identical to `hand`. |

## Scores file (`fingosc_scores.csv`)

Each block produces three columns prefixed with the block key. For the bundled configs that means `dominant_*` and `nondominant_*`.

### Per-block metric set

Computed in `fingosc.score_df` ([mend2np/fingosc.py](../mend2np/fingosc.py)).

| Metric suffix | Type | Description |
| --- | --- | --- |
| `_n_trials` | int | Number of taps recorded in this block. |
| `_mean_rt` | float | Mean inter-tap interval (seconds). |
| `_sd_rt` | float | SD of inter-tap interval. |

### Block slots

The block name is whatever you set as the JSON object key under `blocks`. In the bundled example configs this is:

| Block prefix | Description |
| --- | --- |
| `dominant` | Tap block for the participant's dominant hand. |
| `nondominant` | Tap block for the non-dominant hand. |

So the score columns are: `dominant_n_trials`, `dominant_mean_rt`, `dominant_sd_rt`, `nondominant_n_trials`, `nondominant_mean_rt`, `nondominant_sd_rt`.
