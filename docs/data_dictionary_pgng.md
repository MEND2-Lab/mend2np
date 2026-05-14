# Data dictionary — PGNG

The Parametric Go / No-go / Stop task (PGNG) tests inhibitory control. Stimuli are presented in rapid succession; some stimuli are *targets* requiring a key press, and the rules differ by block:

- **Go (`go`)** — respond to every target.
- **Go / No-go (`gng`)** — respond to a target on its *first* occurrence; withhold the response when the same target appears a second time consecutively (a *lure*).
- **Go / Stop (`gs`)** — respond to every target, *unless* the next stimulus shown is the configured stop signal (in which case withhold).

Each block is configured separately under `params['blocks']`. Block keys typically encode the target count (e.g. `1`, `2`, `3` targets per block); the score column prefix uses that count, e.g. `gng_2T_*` for a 2-target Go/No-go block.

`mend2np.pgng` writes:

- `<exp_name>_n<N>_trials_<timestamp>.csv` — one row per trial across all blocks.
- `<exp_name>_n<N>_scores_<timestamp>.csv` — one row per input file.

## Shared metadata columns (both files)

| Column | Type | Description |
| --- | --- | --- |
| `id` | str | Participant ID. Source: `metacols.id`. |
| `filename` | str | Basename of the input CSV. |
| `session` | str/int | Session number. Source: `metacols.session`. |
| `datetime` | str | Date/time. Source: `metacols.datetime`. |
| `exp_name` | str | Experiment name. Source: `metacols.exp_name`. |
| `software_version` | str | PsychoPy / E-Prime version. Source: `metacols.software_version`. |
| `framerate` | float | Display framerate (Hz). Source: `metacols.framerate`. |
| `os` | str | Operating system. Source: `metacols.os`. |

## Trials file (`pgng_trials.csv`)

One row per trial, with the per-block raw fields renamed to standard names plus several derived classifications.

| Column | Type | Description |
| --- | --- | --- |
| (shared metadata) | — | See block above. |
| `stimuli` | str | Stimulus filename presented on this trial. Source: `blocks.<B>.cols.stimuli`. |
| `response` | str | Raw response key (or NaN if no response). Source: `blocks.<B>.cols.response`. |
| `rt` | float | Response time (seconds) on this row. NaN when no response. Source: `blocks.<B>.cols.rt`. E-Prime values are converted from ms to s. |
| `trial` | float | Trial counter within the block. Source: `blocks.<B>.cols.trial`. |
| `block` | str | Block key from the config (typically the target count: `'1'`, `'2'`, `'3'`). |
| `stim_targ_names` | list[str] | List of target stimulus filenames for this block. Broadcast from `blocks.<B>.metavars.stim_targ_names`. |
| `resp_key` | list[str] | Keyboard key(s) the participant should press for a "go". Broadcast from `blocks.<B>.metavars.resp_key`. |
| `stim_dur` | float | Stimulus duration (seconds). Either static (`blocks.<B>.metavars.stim_dur`) or, for `gs` blocks, taken from `cols.stop_time`. |
| `type` | str | Block type: `'go'`, `'gng'`, or `'gs'`. Broadcast from `blocks.<B>.metavars.type`. |
| `stop_time` | float | (gs blocks only) Per-trial stop-signal time. Source: `blocks.<B>.cols.stop_time`. |
| `stop` | str | (gs blocks only) Filename of the stop stimulus. Broadcast from `blocks.<B>.metavars.stop`. |
| `stim_class` | str | Trial classification: `'target'`, `'lure'`, or `''` (non-target / non-lure). Computed by `pgng.events_df`. |
| `resp_class` | str | Response classification (see table below). Computed by `pgng.events_df`. |
| `rt_adj` | float | Cross-row-adjusted RT (seconds). For responses recorded on the trial after the stimulus (PGNG response windows can spill across rows), this adds the inter-row offset; see `pgng.rt_adj`. |
| `exp_start` | float | Experiment start time. Optional, present only when `metacols.exp_start` is configured. |
| `stim_start` | float | Per-trial stimulus onset time. Optional, present only when `cols.stim_start` is configured (or estimable from `exp_start + start_delta + i*stim_dur`). |
| `stim_start_adj` | float | `stim_start − exp_start`. Present only when both timing columns are available. |
| `onsets` | float | Time at which the participant's response occurred relative to `exp_start`. Falls back to stimulus onset when no RT is available. |

### `resp_class` values

| Value | Where it appears | Meaning |
| --- | --- | --- |
| `hit` | go / gng / gs | Participant responded to a target on the current row or the next row. |
| `om` | go / gng / gs | Omission — target was shown but no response was given. |
| `com` | gng / gs | Commission error — participant responded to a lure (gng: within next row; gs: within next 2 rows). |
| `rej` | gng / gs | Correct rejection — participant withheld response on a lure. |
| `mo` | gng | "Missed-then-omitted" — the previous target was an omission, so behaviour on the lure is uninterpretable. |
| `randcom` | go / gng / gs | Random commission — response on a non-target / non-lure trial that wasn't preceded by a target/lure (gs: also requires the row two back to not be target/lure). |
| `''` (empty) | all | No classifiable event. |

## Scores file (`pgng_scores.csv`)

For each block, the scorer emits a set of metrics prefixed with `<type>_<N>T_` where `<type>` is `go` / `gng` / `gs` and `<N>` is the target count for that block (e.g. `gs_2T_*`).

### Per-block metric set

The columns produced depend on block type:

| Suffix | go | gng | gs | Description |
| --- | :---: | :---: | :---: | --- |
| `_hit` | x | x | x | Count of target hits. |
| `_om` | x | x | x | Count of target omissions. |
| `_com` |  | x | x | Count of lure commission errors. |
| `_rej` |  | x | x | Count of correct lure rejections. |
| `_mo` |  | x |  | Count of `mo` events. |
| `_randcom` | x | x | x | Count of random commissions. |
| `_hit_rt_mean` | x | x | x | Mean `rt_adj` for hits. |
| `_hit_rt_sd` | x | x | x | SD of `rt_adj` for hits. |
| `_com_rt_mean` |  | x | x | Mean `rt_adj` for commissions. |
| `_com_rt_sd` |  | x | x | SD of `rt_adj` for commissions. |
| `_stp_tm_rej` |  |  | x | Mean `stim_dur` (= stop signal time) on correctly-rejected lures. |
| `_stp_tm_com` |  |  | x | Mean `stim_dur` on commission-error lures. |
| `_pctt` | x | x | x | Percent of targets correctly hit (`hit / n_targets`, 0–1). |
| `_pcit` |  | x | x | Percent of inhibition trials correctly rejected (`rej / n_lures`, 0–1). |

### Block prefix slots

The slots depend on which blocks your config defines and how many targets each block has. For the bundled example configs:

| Prefix | Block type | Target count |
| --- | --- | --- |
| `go_2T` | Go | 2 |
| `go_3T` | Go | 3 |
| `gng_2T` | Go / No-go | 2 |
| `gng_3T` | Go / No-go | 3 |
| `gs_2T` | Go / Stop | 2 |
| `gs_3T` | Go / Stop | 3 |

Combine prefix + suffix to get a column name. e.g. `gng_3T_pcit` is the inhibition accuracy on the 3-target Go/No-go block.
