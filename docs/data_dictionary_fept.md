# Data dictionary — FEPT

The Facial Emotion Perception Task (FEPT) presents brief images and asks the participant to identify what was shown. Stimuli are typically grouped into a *faces* block (images of faces with varying emotion, race, sex) and an *animals* block (used as a control). Each stimulus filename encodes the trial's category labels (e.g. `As_F_Hap_152.jpg` → asian / female / happy), and the scorer parses these via per-block regex maps in the JSON config.

`mend2np.fept` writes:

- `<exp_name>_n<N>_trials_<timestamp>.csv` — one row per trial across all input files.
- `<exp_name>_n<N>_scores_<timestamp>.csv` — one row per input file (participant × session).

Optionally, per-input CSVs can also be written via `ind=True`; those mirror the trials shape described below.

## Shared metadata columns

| Column | Type | Description |
| --- | --- | --- |
| `id` | str | Participant ID. Source: `metacols.id`. |
| `filename` | str | Basename of the input CSV. |
| `session` | str/int | Session number. Source: `metacols.session`. |
| `datetime` | str | Date/time the experiment was run. Source: `metacols.datetime`. |
| `exp_name` | str | Experiment name. Source: `metacols.exp_name`. |
| `software_version` | str | PsychoPy / E-Prime version. Source: `metacols.software_version`. |
| `framerate` | float | Display framerate (Hz). Source: `metacols.framerate`. |
| `os` | str | Operating system. Source: `metacols.os`. |

## Trials file (`fept_trials.csv`)

One row per trial across all input files. Columns appear in the order shown.

| Column | Type | Description |
| --- | --- | --- |
| (shared metadata) | — | See block above. |
| `stimuli` | str | Stimulus filename. Source: `blocks.<B>.cols.stimuli`. |
| `response` | str | Raw response key/label. Source: `blocks.<B>.cols.response`. |
| `rt` | float | Response time (seconds), measured from the response window opening. Source: `blocks.<B>.cols.rt`. |
| `correct_response` | str | The expected response key for this stimulus. Source: `blocks.<B>.cols.correct_response`. |
| `type` | str | Block type label, used as the column prefix in the scores file (e.g. `'faces'`, `'animals'`). Source: `blocks.<B>.metavars.type`. |
| `stimulus_duration` | float | Static stimulus duration (seconds). Broadcast from `blocks.<B>.metavars.stimulus_duration`. |
| `mask_duration` | float | Static post-stimulus mask duration (seconds). Broadcast from `blocks.<B>.metavars.mask_duration`. |
| `response_label` | str | Human-readable label for `response` via `blocks.<B>.key_labels` (e.g. `'happy'` from key `'k'`). |
| `correct_response_label` | str | Human-readable label for `correct_response`. |
| `stim_class` | str | `;`-separated list of category labels parsed from the stimulus filename via `blocks.<B>.stim_class_map` (e.g. `'happy;asian;female'`). |
| `block` | str | Block key from the config (the JSON object key under `blocks`). |

## Scores file (`fept_scores.csv`)

The score columns follow a `<type>[_<category>]_<metric>` pattern. `<type>` is the block label (typically `faces` or `animals`); `<category>` is one of the per-block category labels parsed from the stimulus filename via `stim_class_map`.

### Per-block metric set

For each block (and for each category subset within a block), the following nine metrics are written. Computed in `fept.score_blk` ([mend2np/fept.py](../mend2np/fept.py)).

| Metric suffix | Type | Description |
| --- | --- | --- |
| `_correct_ct` | int | Number of trials where `response == correct_response`. |
| `_incorrect_ct` | int | Number of trials where `response != correct_response`. |
| `_acc` | float | `correct_ct / (correct_ct + incorrect_ct)`. NaN if the bucket is empty. |
| `_rt_mean` | float | Mean RT (seconds) across all trials in the bucket. |
| `_rt_sd` | float | SD of RT across all trials in the bucket. |
| `_correct_rt_mean` | float | Mean RT restricted to correct trials. |
| `_correct_rt_sd` | float | SD of RT restricted to correct trials. |
| `_incorrect_rt_mean` | float | Mean RT restricted to incorrect trials. |
| `_incorrect_rt_sd` | float | SD of RT restricted to incorrect trials. |

### Block-level columns

| Pattern | Description |
| --- | --- |
| `<type>_<metric>` | Bucket = every trial of the block. e.g. `faces_acc`, `animals_rt_mean`. `<type>` is whatever you set in `blocks.<B>.metavars.type` (typical: `faces`, `animals`). |

### Per-category columns

For each block, the `stim_class` field is split on `;` into one column per category; each category value becomes its own per-bucket score group. Slot enumeration depends on what your `stim_class_map` produces; the ones in the example data are:

| Block | Category dimension | Slot values seen in example data |
| --- | --- | --- |
| `faces` | emotion | `angry`, `fear`, `happy`, `neutral`, `sad` |
| `faces` | race | `asian`, `black`, `white` |
| `faces` | sex | `female`, `male` |
| `animals` | species | `bird`, `cat`, `cow`, `dog`, `fish` |

| Pattern | Description |
| --- | --- |
| `<type>_<category>_<metric>` | Bucket = trials whose stimulus matched this category label. e.g. `faces_happy_acc`, `animals_dog_rt_mean`, `faces_female_correct_rt_sd`. |

### Misclassification counts

For each correct-response category, the scorer tallies how often the participant gave each *wrong* response label.

| Pattern | Description |
| --- | --- |
| `<type>_<correct_label>_as_<wrong_label>` | Count of trials whose true label was `<correct_label>` but the participant pressed the key for `<wrong_label>`. e.g. `faces_angry_as_neutral` = how many angry-face trials were labelled neutral. `<wrong_label>` is `nan` when the participant did not respond. These columns are **dynamic** — only label-pair combinations that actually occurred in the data appear in the output. |

### False-alarm counts

| Pattern | Description |
| --- | --- |
| `<type>_<response_label>_false_alarm` | Count of trials where the participant gave `<response_label>` even though `<response_label>` was not the correct label (across all trials of the block). e.g. `faces_neutral_false_alarm` = total neutral-key presses on non-neutral trials. Also dynamic; only labels actually given as wrong responses appear. |
