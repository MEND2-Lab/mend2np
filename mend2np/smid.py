'''
SMID (Social Monetary Incentive Delay) scoring.

The Social Monetary Incentive Delay task probes how reward and loss incentives
modulate response readiness, separately for self-benefiting vs. charity-
benefiting outcomes. Each trial:

  1. A *prime* image is shown indicating the outcome at stake — gain / lose /
     neither, and small ($0.20) / big ($5) / zero amount.
  2. After a delay there is a brief *probe* window. The participant must
     respond during the probe to earn the gain (or avoid the loss). Responding
     too early "spoils" the trial; not responding misses it.
  3. The trial is labelled `benefactor=YOURSELF` (non-social) or
     `benefactor=NAME` (social — outcome accrues to the charity the
     participant selected).

Participants pre-select a charity (`charity_name`) and a representative
(`staff_name`); both are carried through into the trial-level and score
outputs so downstream analyses can link conditions to who was benefiting.

This module supports a *practice* block and a *real* block, each with its own
column-name set (the practice columns are prefixed with `p`). All four
example CSVs include both — sometimes the real block has only a handful of
trials (incomplete runs); the user explicitly wanted those preserved in
output, so this module never filters by trial count.
'''

import logging
import os
import re
import pandas as pd
import numpy as np
from mend2np.utils import (
    setup_logger,
    get_meta_cols,
    handle_multiple_responses,
    validate_params,
    run_task,
    copy_configured_columns,
)

logger = logging.getLogger('root')

REQUIRED_PARAMS = {
    'metacols': dict,
    'blocks': dict,
}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def smid(params:dict, out:str=os.getcwd(), write:bool=True, filelist:str|list='',
         formatted:bool=False, log=20, trial_filter:str='') -> tuple:
    """Score one or more SMID data files.

    :param params: configuration dict. Must include `metacols` and `blocks`.
        See `tests/smid_example.json`.
    :param out: directory for output CSVs.
    :param write: if True, write combined trials + scores CSVs.
    :param filelist: list of CSV paths, path to a file-of-paths, or empty (GUI picker).
    :param formatted: True if data are already tidy with standard column names.
    :param log: log level.
    :param trial_filter: optional pandas query string applied before scoring.
    :returns: `(combined_scores, combined_trials)`.
    """
    setup_logger(name='root', out=out, level=log).info('start')
    validate_params(params, REQUIRED_PARAMS)

    def process_one(filepath, params, logger):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        if not formatted:
            df = format_df(df, params)
        df.insert(1, 'filename', filename)
        scores_row = pd.concat([get_meta_cols(df, params), score_df(df, trial_filter)], axis=1)
        scores_row.insert(1, 'filename', filename)
        return df, scores_row

    return run_task(
        params=params, filelist=filelist, out=out, write=write,
        process_file_fn=process_one,
    )


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

# Prime-image filename → (reward_type, amount, amount_label). The amount_label
# is what's used in score column names (e.g. "real_self_gain_big_n_probes").
_PRIME_PATTERN = re.compile(r'(win|lose|neutral)\s*(small|big)?', re.IGNORECASE)
_AMOUNT_LABELS = {0.0: 'zero', 0.2: 'small', 5.0: 'big'}


def parse_prime(value) -> dict:
    """Decode a SMID prime image filename into `(reward_type, amount, amount_label)`.

    Examples:
      `images/WinSmall.jpg`   -> ('gain',    0.2, 'small')
      `images/LoseBig.jpg`    -> ('lose',    5.0, 'big')
      `images/Neutral.jpg`    -> ('neither', 0.0, 'zero')
      NaN or unrecognised     -> (None, np.nan, None)
    """
    if not isinstance(value, str):
        return {'reward_type': None, 'amount': np.nan, 'amount_label': None}
    match = _PRIME_PATTERN.search(value)
    if not match:
        return {'reward_type': None, 'amount': np.nan, 'amount_label': None}
    kind = match.group(1).lower()
    size = (match.group(2) or '').lower()
    if kind == 'neutral':
        return {'reward_type': 'neither', 'amount': 0.0, 'amount_label': 'zero'}
    reward_type = 'gain' if kind == 'win' else 'lose'
    amount = 0.2 if size == 'small' else 5.0 if size == 'big' else np.nan
    label = _AMOUNT_LABELS.get(amount)
    return {'reward_type': reward_type, 'amount': amount, 'amount_label': label}


def format_df(df:pd.DataFrame, params:dict) -> pd.DataFrame:
    """Reshape a raw SMID CSV into the library's standard column layout.

    For each `params['blocks']` entry (typically `practice` and `real`):
      - Mask to rows where this block's `trial` column is non-null.
      - Copy the configured per-block columns onto a tmp dataframe.
      - Parse list-string RT cells back into scalars and coerce to numeric.
      - Decode the `prime` image filename into `reward_type`, `amount`,
        `amount_label`, and decode `benefactor` into a boolean `social` flag.
      - Compute a `correct` flag — `probe_response` if present, otherwise
        `probe_rt.notna()` (a non-null RT means they responded in the window).
      - Broadcast `metavars` (e.g. `phase: 'practice'`) and the per-participant
        metacols onto every row, and stamp the block key as `block`.
    Returns the concatenated per-block dataframes.
    """
    fmtdf = pd.DataFrame()

    for block_key, block_cfg in params['blocks'].items():
        if isinstance(block_key, str) and block_key.startswith('_'):
            continue

        block_cols = block_cfg.get('cols', {})
        block_metavars = block_cfg.get('metavars', {})
        trial_col = block_cols.get('trial')

        # Mask: rows where this block's trial counter is non-null. Same pattern as pgng.
        if not trial_col or trial_col not in df.columns:
            logger.warning(
                f"blocks.{block_key}: trial column "
                f"'{trial_col}' is missing; skipping this block."
            )
            continue
        mask = df[trial_col].notna()
        if mask.sum() == 0:
            logger.info(f"blocks.{block_key}: 0 trial rows; skipping.")
            continue

        tmpdf = pd.DataFrame()
        copy_configured_columns(
            tmpdf, df, block_cols, f'blocks.{block_key}.cols',
            mask=mask, logger=logger,
        )

        # Parse list-string cells. The RT/key response columns are usually written
        # as PsychoPy list-reprs like `[0.43]` even though there's only one entry.
        for resp_col in ('probe_key', 'probe_rt', 'pre_key', 'post_probe_key'):
            if resp_col in tmpdf.columns:
                tmpdf[resp_col] = tmpdf[resp_col].apply(
                    lambda x: handle_multiple_responses(x, slice_index=0)
                )

        # Coerce probe_rt to numeric. `handle_multiple_responses` returns the first
        # element of the list; for empty `[]` it returns the empty list (a non-numeric
        # value), so `to_numeric(errors='coerce')` cleans those to NaN.
        if 'probe_rt' in tmpdf.columns:
            tmpdf['probe_rt'] = pd.to_numeric(tmpdf['probe_rt'], errors='coerce')

        # Decode the prime image into reward_type / amount / amount_label.
        if 'prime' in tmpdf.columns:
            parsed = tmpdf['prime'].apply(parse_prime).apply(pd.Series)
            tmpdf['reward_type'] = parsed['reward_type']
            tmpdf['amount'] = parsed['amount']
            tmpdf['amount_label'] = parsed['amount_label']

        # Decode the benefactor into a boolean social flag.
        # 'NAME' (placeholder for the charity rep) → social trial.
        # 'YOURSELF' → non-social (self) trial.
        if 'benefactor' in tmpdf.columns:
            ben = tmpdf['benefactor'].astype(str).str.upper()
            tmpdf['social'] = ben.eq('NAME')
            tmpdf['social_label'] = np.where(tmpdf['social'], 'charity', 'self')

        # Correctness — prefer the most direct signal available:
        #   1. explicit `probe_response` boolean (real trials in most variants),
        #   2. `probe_rt` non-null (a non-null RT means the participant responded
        #      in the probe window, which is the definition of correct here),
        #   3. `feedback_correct` non-null (older CSVs that don't record a
        #      probe RT still record which feedback string was shown).
        if 'probe_response' in tmpdf.columns:
            tmpdf['correct'] = tmpdf['probe_response'].astype(bool)
        elif 'probe_rt' in tmpdf.columns:
            tmpdf['correct'] = tmpdf['probe_rt'].notna()
        elif 'feedback_correct' in tmpdf.columns:
            tmpdf['correct'] = tmpdf['feedback_correct'].notna()

        # Static per-block metadata (e.g. phase: 'practice').
        for metavar, value in block_metavars.items():
            if isinstance(metavar, str) and metavar.startswith('_'):
                continue
            tmpdf[metavar] = value

        # Per-participant metacols broadcast across every row (deferred to after
        # tmpdf has rows; see the pgng.format_df comment for why this matters).
        for metacol in params['metacols']:
            if isinstance(metacol, str) and metacol.startswith('_'):
                continue
            csv_col = params['metacols'][metacol]
            if not csv_col:
                continue
            if csv_col in df.columns:
                idx = df[csv_col].first_valid_index()
                tmpdf[metacol] = df.at[idx, csv_col] if idx is not None else None
            else:
                logger.warning(
                    f"metacols.{metacol}: configured CSV column '{csv_col}' is not "
                    f"in this file's columns — '{metacol}' will be missing for "
                    f"block {block_key}."
                )

        # Reorder: metacols first (matches the other tasks' layout).
        metacol_names = [m for m in params['metacols'] if m in tmpdf.columns
                         and not (isinstance(m, str) and m.startswith('_'))]
        other_cols = [c for c in tmpdf.columns if c not in metacol_names]
        tmpdf = tmpdf[metacol_names + other_cols]

        tmpdf['block'] = block_key
        fmtdf = pd.concat([fmtdf, tmpdf], ignore_index=True)

    return fmtdf


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _condition_label(social_label, reward_type, amount_label) -> str | None:
    """Build a stable column-name fragment for a (social, type, amount) bucket.

    `neither` trials always have amount=0, so the amount suffix is suppressed for
    those — yielding shorter, more readable names like `self_neither_n_probes`
    instead of `self_neither_zero_n_probes`. Returns None if any input is missing.
    """
    if not social_label or not reward_type or not amount_label:
        return None
    if reward_type == 'neither':
        return f'{social_label}_neither'
    return f'{social_label}_{reward_type}_{amount_label}'


def score_df(df:pd.DataFrame, trial_filter:str) -> pd.DataFrame:
    """Build a 1-row dataframe of per-file SMID scores.

    Practice trials are excluded from scoring (rows where `phase == 'practice'`
    are dropped). Practice trials still appear in the trials-level output — they
    just don't contribute to the aggregate scores.

    For the remaining trials, groups by `social_label` (self / charity) ×
    `reward_type` (gain / lose / neither) × `amount_label` (small / big /
    zero). For each bucket, emits four columns:

      `<condition>_n_probes`
      `<condition>_prop_correct`
      `<condition>_mean_rt`
      `<condition>_sd_rt`

    where `<condition>` is `<social>_<type>_<amount>` (with the `_<amount>`
    suffix suppressed for `neither` trials, which always have amount=0).
    The block name is not used as a prefix — once practice trials are filtered
    out only the single "real" block typically remains, so the prefix would
    just be noise. If a config has multiple non-practice blocks the same
    condition columns will collide across blocks, in which case revisit.

    :param df: per-trial dataframe (produced by format_df).
    :param trial_filter: optional pandas query string applied before scoring.
    """
    # Drop practice rows. The `phase` column is broadcast in `format_df` from
    # the per-block `metavars.phase` config setting.
    if 'phase' in df.columns:
        df = df.loc[df['phase'] != 'practice']

    if trial_filter:
        df = df.query(trial_filter)

    score_dict:dict = {}
    needed = {'social_label', 'reward_type', 'amount_label', 'correct'}
    if not needed.issubset(df.columns):
        # Not enough condition info to score — just return the empty frame so
        # the metacols still come through in the merged output.
        return pd.DataFrame(score_dict, index=[0])

    grouped = df.groupby(
        ['social_label', 'reward_type', 'amount_label'],
        dropna=False,  # keep buckets that landed in NaN levels (e.g. unparseable prime)
    )
    for (social_label, reward_type, amount_label), bucket in grouped:
        condition = _condition_label(social_label, reward_type, amount_label)
        if condition is None:
            # One of the dimensions was NaN — record under a clearly-labelled key
            # so the user can see that a condition couldn't be derived.
            condition = 'unknown'
        prefix = condition
        score_dict[f'{prefix}_n_probes'] = len(bucket)
        score_dict[f'{prefix}_prop_correct'] = bucket['correct'].astype(float).mean()
        # `probe_rt` is NaN for missed/spoiled trials, so .mean()/.std() automatically
        # restrict to the correct-response RTs — which is the user-requested behaviour.
        if 'probe_rt' in bucket.columns:
            score_dict[f'{prefix}_mean_rt'] = bucket['probe_rt'].mean()
            score_dict[f'{prefix}_sd_rt'] = bucket['probe_rt'].std()

    return pd.DataFrame(score_dict, index=[0])
