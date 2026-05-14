# Data dictionary — BART

The Balloon Analogue Risk Task (BART) is a behavioural measure of risk-taking. On each trial the participant repeatedly pumps a balloon to inflate it; each pump adds money to a temporary bank but also raises the chance the balloon will pop. The participant chooses when to bank the money (cashing out the trial's earnings) versus continuing to pump (risking a pop, which forfeits the trial's earnings).

This document describes every column in the two output files produced by `mend2np.bart`:

- `<exp_name>_n<N>_trials_<timestamp>.csv` — one row per trial.
- `<exp_name>_n<N>_scores_<timestamp>.csv` — one row per input file (participant × session).

## Shared metadata columns (both files)

These appear at the start of every row and are copied through from the input CSV via the `metacols` config section. See [`CONFIGURING.md`](CONFIGURING.md) for how to map them.

| Column | Type | Description |
| --- | --- | --- |
| `id` | str | Participant ID. Source: `metacols.id`. |
| `filename` | str | Basename of the input CSV (added by the scorer; not user-configured). |
| `session` | str/int | Session number. Source: `metacols.session`. |
| `datetime` | str | Date/time the experiment was run. Source: `metacols.datetime`. |
| `exp_name` | str | Experiment name string from the source CSV. Source: `metacols.exp_name`. |
| `software_version` | str | PsychoPy / E-Prime version. Source: `metacols.software_version`. |
| `framerate` | float | Display framerate (Hz). Source: `metacols.framerate`. |
| `os` | str | Operating system the data were collected on. Source: `metacols.os`. |

## Trials file (`bart_trials.csv`)

One row per balloon trial.

| Column | Type | Description |
| --- | --- | --- |
| (shared metadata) | — | See block above. |
| `nPumps` | float | Number of pumps the participant made on this balloon. Source: `cols.nPumps`. |
| `popped` | bool | True if the balloon popped (risk realised); False if the participant successfully banked. Source: `cols.popped`, coerced to bool. |
| `earnings` | float | Money earned on this trial (0 if popped). Source: `cols.earnings`. |
| `trial` | float | Trial counter (0-indexed). Source: `cols.trial`. |
| `rt` | list[float] | Inter-pump latencies in seconds. The raw CSV stores a list of cumulative click timestamps; the scorer converts these to *deltas between successive clicks*, so the first element is the latency from balloon onset to the first pump and subsequent elements are pump-to-pump latencies. Empty list if the participant did not pump. Source: `cols.rt`. |

## Scores file (`bart_scores.csv`)

One row per input file. Computed in `bart.score_df` ([mend2np/bart.py](../mend2np/bart.py)).

| Column | Type | Description |
| --- | --- | --- |
| (shared metadata) | — | See block above. |
| `ntrials_popped` | int | Count of trials where the balloon popped. |
| `ntrials_unpopped` | int | Count of trials where the participant banked successfully. |
| `popped_ratio` | float | `ntrials_popped / ntrials_unpopped`. NaN when no banked trials. |
| `ptrials_popped` | float | Proportion of trials that popped, 0–1. |
| `ptrials_unpopped` | float | Proportion of trials banked, 0–1. |
| `mean_pumps_popped` | float | Mean `nPumps` across popped trials. |
| `mean_pumps_unpopped` | float | Mean `nPumps` across banked trials. |
| `mean_rt_unpopped` | float | Mean inter-pump latency (seconds) across all pumps on banked trials. |
| `sd_rt_unpopped` | float | Standard deviation of inter-pump latency on banked trials. |
| `total_earnings` | float | Sum of `earnings` across all trials. |
| `mean_earnings` | float | Mean `earnings` across all trials. |
| `intertrial_variability` | float | `SD(nPumps) / mean(nPumps)` across all trials — a normalised risk-variability measure (coefficient of variation). |
| `post_failure_mean_pumps` | float | Mean `nPumps` on the trial *immediately following* a popped trial. Indexes post-failure caution. |
| `post_failure_mean_rt` | float | Mean inter-pump latency on the trial after a pop. |
| `post_failure_sd_rt` | float | SD of inter-pump latency on the trial after a pop. |
| `post_pumps_loss` | float | Mean of `nPumps_t − nPumps_{t+1}` evaluated only when trial *t* popped and trial *t+1* was banked. Captures how much the participant pulled back after losing. |
