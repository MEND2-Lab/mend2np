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

Response detail is captured for all three routines of a trial — `pre_probe_*` (early responses), `probe_*` (in-window responses), and `post_probe_*` (late responses). Older PsychoPy exports sometimes ship the `.keys` columns without the matching `.rt` columns; when an RT column is absent the run logs a warning naming the config path and the column stays blank, while the response flag still resolves from the key column.

#### A note on RT clocks

PsychoPy logs every keyboard component's RT from **its own routine's onset**, and all three RT columns are preserved exactly that way — nothing is silently rebased onto a common clock. So each column answers a question about its own routine:

| Column | Measured from | Reads as |
| --- | --- | --- |
| `pre_probe_rt` | variable-delay onset | how far into the delay the participant jumped the gun |
| `probe_rt` | probe onset | the response time of interest |
| `post_probe_rt` | post-probe (jitter) onset | how soon after probe offset the late response arrived |

**The three are not directly comparable to each other.** If you need a common timeline, build it explicitly — the onset columns are in the output for exactly this purpose. Late responses relative to probe onset are `post_probe_rt + probe_duration_realized`; that puts them just after probe offset (0.20–0.90 s in the bundled examples).

Early responses cannot be put on the probe clock at all, and this is a property of the task rather than the export. An early keypress ends the variable-delay routine, so on exactly those trials the probe starts about one frame later instead of at its scheduled time. Rebasing would report roughly −0.01 s for every early response regardless of whether the participant pressed 70 ms or 700 ms into the delay, and the scheduled onset cannot be recovered (no scheduled-delay column is written, and `variable_delay.stopped` is truncated by the press).

### `probe_duration` vs `probe_duration_realized`

These measure different things and diverge systematically, so pick deliberately:

| Trials | Relationship (bundled examples) |
| --- | --- |
| No in-window response (n=71) | `realized ≈ requested`, within ±17 ms — frame quantization only. |
| With an in-window response (n=132) | The response changes when the routine ends: `realized` runs past `probe_rt` by 19–366 ms (median 84 ms) and usually falls *short* of the requested duration. |
| Practice (n=18) | No requested duration exists; `realized` has a median of 0.48 s and reaches 3.98 s. |

Use `probe_duration` for the stimulus parameter the task intended to present (it is adaptive across trials — 0.21 s to 0.43 s in the examples). Use `probe_duration_realized` for the actual elapsed time on screen, and as the offset to add to `post_probe_rt` when placing late responses on the probe clock. Do not treat `probe_duration_realized` as a measurement error around `probe_duration` on response trials — the gap there is task behaviour, not timing jitter.

`pre_probe_rt` is deliberately **left on the delay clock** and is *not* comparable to the other two. The reason is a property of the task: an early keypress ends the variable-delay routine, so on exactly those trials the probe starts about one frame later instead of at its scheduled time. Rebasing would report roughly −0.01 s for every early response regardless of whether the participant pressed 70 ms or 700 ms into the delay, and the scheduled onset cannot be recovered from the export (no scheduled-delay column is written, and `variable_delay.stopped` is truncated by the press). As logged, `pre_probe_rt` answers the question the data can actually support — how far into the delay the participant jumped the gun.

| Column | Type | Description |
| --- | --- | --- |
| `trial` | float | Trial counter (per block). Source: `blocks.<B>.cols.trial`. |
| `benefactor` | str | `'YOURSELF'` (self trial) or `'NAME'` (charity trial). Source: `blocks.<B>.cols.benefactor`. |
| `prime` | str | Prime image filename — encodes reward type and amount. Source: `blocks.<B>.cols.prime`. |
| `probe_key` | str | Key (or shape name) the participant pressed during the probe window. Source: `blocks.<B>.cols.probe_key`. |
| `pre_probe_key` | str | Key pressed before the probe window opened (early/spoiled trial signal). Source: `blocks.<B>.cols.pre_probe_key`. |
| `pre_probe_rt` | float | Early-response time (seconds) measured from **variable-delay onset**, as logged. Not comparable to `probe_rt` / `post_probe_rt` — see the RT-clocks note above. NaN when no early response was recorded. Source: `blocks.<B>.cols.pre_probe_rt`. |
| `post_probe_key` | str | Key pressed after the probe window closed. Source: `blocks.<B>.cols.post_probe_key`. |
| `post_probe_rt` | float | Late-response time (seconds) measured from **post-probe (jitter) routine onset**, as logged — i.e. how soon after probe offset the response arrived. Not comparable to `probe_rt` without adding `probe_duration_realized`. NaN when no late response was recorded. Source: `blocks.<B>.cols.post_probe_rt`. |
| `probe_onset` | float | Probe routine onset on the global experiment clock (seconds). Emitted so callers can place events on a common timeline. Source: `blocks.<B>.cols.probe_onset`. |
| `post_probe_onset` | float | Post-probe (jitter) routine onset on the global experiment clock (seconds). Source: `blocks.<B>.cols.post_probe_onset`. |
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
| `probe_duration_realized` | float | How long the probe routine **actually ran** (`post_probe_onset - probe_onset`), in seconds. This is not simply a frame-quantized `probe_duration` — see the note below. The only probe-duration measure available on practice trials, which have no requested-duration column. |
| `correct` | bool | Trial-level correctness. Derived in priority order: `probe_response` if present → `probe_rt.notna()` → `feedback_correct.notna()`. |
| `pre_probe_response` | bool | True when an early response was logged before the probe window opened (a spoiled trial). Derived in priority order: `pre_probe_key.notna()` → `pre_probe_rt.notna()`. Does not feed `correct` or the scores file. |
| `post_probe_response` | bool | True when a late response was logged after the probe window closed. Derived in priority order: `post_probe_key.notna()` → `post_probe_rt.notna()`. Always False on practice rows — the practice block has no post-probe response logger of its own and shares the real block's column. Does not feed `correct` or the scores file. |
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
