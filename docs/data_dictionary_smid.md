# Data dictionary — SMID

The Social Monetary Incentive Delay task (SMID) probes how reward and loss incentives modulate response readiness, separately for self-benefiting versus charity-benefiting outcomes. Each trial:

1. A *prime* image indicates the outcome at stake — gain / lose / neither × small ($0.20) / big ($5).
2. After a delay, a brief *probe* window opens. The participant must respond during that window to earn the gain (or avoid the loss). Responding too early "spoils" the trial; not responding misses it.
3. The trial is labelled `benefactor=YOURSELF` (non-social — outcome accrues to the participant) or `benefactor=NAME` (social — outcome accrues to the participant's pre-selected charity).

`mend2np.smid` writes:

- `<exp_name>_n<N>_trials_<timestamp>.csv` — one row per trial across practice + real blocks.
- `<exp_name>_n<N>_scores_<timestamp>.csv` — one row per input file, **practice trials excluded**.

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
| `charity_name` | str | The charity the participant chose. Source: `metacols.charity_name`. Carried into both outputs. |
| `staff_name` | str | The charity representative the participant chose. Source: `metacols.staff_name`. |

## Trials file (`smid_trials.csv`)

One row per trial, including practice trials.

### Configured per-block columns

| Column | Type | Description |
| --- | --- | --- |
| `trial` | float | Trial counter (per block). Source: `blocks.<B>.cols.trial`. |
| `benefactor` | str | `'YOURSELF'` (self trial) or `'NAME'` (charity trial). Source: `blocks.<B>.cols.benefactor`. |
| `prime` | str | Prime image filename — encodes reward type and amount. Source: `blocks.<B>.cols.prime`. |
| `probe_key` | str | Key (or shape name) the participant pressed during the probe window. Source: `blocks.<B>.cols.probe_key`. |
| `pre_key` | str | Key pressed before the probe window opened (early/spoiled trial signal). Source: `blocks.<B>.cols.pre_key`. |
| `post_probe_key` | str | Key pressed after the probe window closed. Source: `blocks.<B>.cols.post_probe_key`. |
| `feedback_correct` | str | Feedback string shown when the trial was correct (typically the "you win" / "you avoided losing" message). Source: `blocks.<B>.cols.feedback_correct`. |
| `feedback_incorrect` | str | Feedback string shown when the trial was missed. Source: `blocks.<B>.cols.feedback_incorrect`. |
| `feedback_spoiled` | str | Feedback string shown when the trial was spoiled (early response). Source: `blocks.<B>.cols.feedback_spoiled`. |
| `probe_duration` | float | Probe window duration (seconds). Source: `blocks.<B>.cols.probe_duration`. |
| `self_earnings` | float | Per-trial self earnings tally. Source: `blocks.<B>.cols.self_earnings`. |
| `charity_earnings` | float | Per-trial charity earnings tally. Source: `blocks.<B>.cols.charity_earnings`. |
| `probe_rt` | float | Probe response time (seconds). NaN for missed/spoiled trials. Source: `blocks.<B>.cols.probe_rt`. |

### Derived columns from `parse_prime` / `format_df`

| Column | Type | Description |
| --- | --- | --- |
| `reward_type` | str | `'gain'` (Win-prefixed prime), `'lose'` (Lose-prefixed prime), `'neither'` (Neutral prime). |
| `amount` | float | Stake in dollars: `0.2` (small), `5.0` (big), `0.0` (neither). |
| `amount_label` | str | Human label: `'small'` / `'big'` / `'zero'`. |
| `social` | bool | True when `benefactor == 'NAME'` (charity trial); False when `'YOURSELF'`. |
| `social_label` | str | `'charity'` when `social` is True, else `'self'`. |
| `correct` | bool | Trial-level correctness. Derived in priority order: `probe_response` if present → `probe_rt.notna()` → `feedback_correct.notna()`. |
| `phase` | str | Block phase, broadcast from `blocks.<B>.metavars.phase`. Trials with `phase == 'practice'` are dropped from the scores file. |
| `block` | str | The block key from the config (typically `'practice'` or `'real'`). |
| `probe_response` | bool | Explicit correctness boolean for real trials when the CSV ships one. Source: `blocks.<B>.cols.probe_response`. May be empty/NaN on practice trials. |

## Scores file (`smid_scores.csv`)

Practice trials are dropped before scoring (rows with `phase == 'practice'`). The block name is **not** used as a column prefix — the bundled configs leave only one non-practice block (`real`) after filtering.

### Per-bucket metric set

For each `(social_label, reward_type, amount_label)` bucket, four columns are emitted. Computed in `smid.score_df` ([mend2np/smid.py](../mend2np/smid.py)).

| Metric suffix | Type | Description |
| --- | --- | --- |
| `_n_probes` | int | Trial count in the bucket. |
| `_prop_correct` | float | Mean of `correct` in the bucket. |
| `_mean_rt` | float | Mean of `probe_rt` in the bucket. Because `probe_rt` is NaN for missed/spoiled trials, this is effectively the mean of correct-response RTs. |
| `_sd_rt` | float | SD of `probe_rt` in the bucket. |

### Bucket prefix patterns

| Pattern | Description |
| --- | --- |
| `<social>_<reward>_<amount>_<metric>` | Three-axis bucket: social ∈ {`self`, `charity`}, reward ∈ {`gain`, `lose`}, amount ∈ {`small`, `big`}. e.g. `charity_gain_big_mean_rt`. |
| `<social>_neither_<metric>` | Special case for `reward_type == 'neither'`: amount is suppressed because all neither trials carry amount `0.0`. e.g. `self_neither_prop_correct`. |
| `unknown_<metric>` | Any bucket where one or more grouping dimensions could not be derived (e.g. an unparseable `prime`). Surfaced explicitly so unmapped trials are visible in the scores file. |

### Bucket slot enumeration

Combining the axes gives this set of expected buckets:

| `social_label` | `reward_type` | `amount_label` | Resulting prefix |
| --- | --- | --- | --- |
| `self` | `gain` | `small` | `self_gain_small` |
| `self` | `gain` | `big` | `self_gain_big` |
| `self` | `lose` | `small` | `self_lose_small` |
| `self` | `lose` | `big` | `self_lose_big` |
| `self` | `neither` | (suppressed) | `self_neither` |
| `charity` | `gain` | `small` | `charity_gain_small` |
| `charity` | `gain` | `big` | `charity_gain_big` |
| `charity` | `lose` | `small` | `charity_lose_small` |
| `charity` | `lose` | `big` | `charity_lose_big` |
| `charity` | `neither` | (suppressed) | `charity_neither` |

Buckets only appear in the output when the participant had at least one trial of that condition.
