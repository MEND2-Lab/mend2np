# Data dictionary — Synonyms

The Synonyms task is a multiple-choice vocabulary test: the participant sees a target word and several option words, and selects the option that means the same as the target. Responses are mapped through `resp_mapping` to a 1–4 option-int namespace so keyboard and touch responses score identically.

`mend2np.synonyms` writes:

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

## Trials file (`synonyms_trials.csv`)

One row per trial.

### Configured columns

| Column | Type | Description |
| --- | --- | --- |
| `correct_resp` | int | The option int that is the correct synonym (mapped through `resp_mapping`). Source: `cols.correct_resp`. |
| `response` | list | List of response tokens recorded on the trial, mapped to option ints. Source: `cols.response`. |
| `rt` | list[float] | List of RTs (seconds), one per response. Source: `cols.rt`. |
| `trial` | float | Trial counter. Source: `cols.trial`. |
| `stimuli` | str | Stimulus filename for the target word. Source: `cols.stimuli`. |
| `stim_dur` | float | Stimulus / response window duration (seconds). Source: `cols.stim_dur`. |

### Derived columns from `parse_responses`

| Column | Type | Description |
| --- | --- | --- |
| `num_responses` | float | Number of responses recorded on the trial. |
| `response_last` | float | Last response (option int). For self-corrected trials this is the participant's final answer. |
| `rt_last` | float | RT of the last response (seconds). Used for scoring. |
| `correct` | float | 1.0 if `correct_resp` appears anywhere in the response list, else 0.0. |
| `correct_resp_index` | float | Index within the response list at which `correct_resp` first appeared (NaN if never). |

## Scores file (`synonyms_scores.csv`)

One row per input file. Computed in `synonyms.score_df` ([mend2np/synonyms.py](../mend2np/synonyms.py)).

| Column | Type | Description |
| --- | --- | --- |
| `num_correct` | int | Total trials where `correct == 1`. |
| `prop_correct` | float | Mean of `correct` (overall accuracy, 0–1). |
| `mean_rt` | float | Mean of `rt_last` across all trials (seconds). |
| `sd_rt` | float | SD of `rt_last` across all trials. |
| `mean_correct_resp_rt` | float | Mean of `rt_last` restricted to correct trials. |
| `std_correct_resp_rt` | float | SD of `rt_last` restricted to correct trials. |
| `mean_incorrect_resp_rt` | float | Mean of `rt_last` restricted to incorrect trials. |
| `std_incorrect_resp_rt` | float | SD of `rt_last` restricted to incorrect trials. |
