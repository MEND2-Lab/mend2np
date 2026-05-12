'''
Stroop task scoring (classic + emotional variants interleaved).

Each trial shows a word in one of three colours; the participant must respond
based on the *colour* the word is rendered in (not the word's meaning). Two
test types alternate in randomised blocks within the same data file:

  - **classic**: colour-word combinations are either `congruent` (e.g. the
    word "BLUE" rendered in blue), `incongruent` (e.g. "BLUE" rendered in
    pink), or `neutral` (XXX in some colour).
  - **emotional**: a word with affective valence (`negative` / `positive`) or
    a `neutral` word is rendered in one of the three colours.

Per-trial outputs include the standardized response option (1 / 2 / 3) so that
keyboard responses (`j`/`k`/`l`) and touch responses
(`trial_opt1_shape`/`trial_opt2_shape`/`trial_opt3_shape`) share a single
integer namespace — the same approach the Synonyms scorer uses.

Scoring is grouped by `test × condition`:
  - number of trials
  - proportion correct
  - mean & SD response time for correct trials
  - mean & SD response time for incorrect trials

Three derived Stroop-interference scores are also produced when the relevant
buckets are present in the data:
  - `classic_stroop_interference_rt` — classic incongruent vs. congruent
  - `emotional_negative_interference_rt` — emotional negative vs. neutral
  - `emotional_positive_interference_rt` — emotional positive vs. neutral
'''

import logging
import os
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
    'cols': dict,
}

# Fallback when the JSON config doesn't supply `resp_mapping`. Covers the
# three-option keyboard layout (j/k/l) plus the practice-block and trial-block
# touch shape names emitted by PsychoPy in the bundled example data.
DEFAULT_RESP_MAPPING = {
    'j': 1,
    'k': 2,
    'l': 3,
    'trial_opt1_shape': 1,
    'trial_opt2_shape': 2,
    'trial_opt3_shape': 3,
    'prac_opt1_shape': 1,
    'prac_opt2_shape': 2,
    'prac_opt3_shape': 3,
}


def _get_resp_mapping(params:dict) -> dict:
    """Return the response → option-index mapping for this run.

    Prefers `params['resp_mapping']` when supplied (so future task variants can
    swap key letters or button labels without touching the code); falls back to
    `DEFAULT_RESP_MAPPING` and logs a warning when absent.
    """
    mapping = params.get('resp_mapping')
    if mapping is None:
        logger.warning(
            "stroop: no 'resp_mapping' in params; falling back to DEFAULT_RESP_MAPPING. "
            "Adding 'resp_mapping' to your JSON config is recommended so the mapping "
            "is explicit per experiment."
        )
        return DEFAULT_RESP_MAPPING
    return mapping


def stroop(params:dict, out:str=os.getcwd(), write:bool=True, filelist:str|list='',
           formatted:bool=False, log=20, trial_filter:str='') -> tuple:
    """Score one or more Stroop data files.

    :param params: configuration dict. Must include `metacols` and `cols`. Should
        also include `resp_mapping` (response label → option int) and
        `color_correct_mapping` (stimulus colour → correct option int). See
        `tests/stroop_example.json`.
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
    resp_mapping = _get_resp_mapping(params)
    color_correct_mapping = params.get('color_correct_mapping', {})
    if not color_correct_mapping:
        logger.warning(
            "stroop: no 'color_correct_mapping' in params — correctness cannot be "
            "derived from `this_color`. Add a mapping like "
            "{'blue': 1, 'saddlebrown': 2, 'hotpink': 3} to score correct/incorrect."
        )

    def process_one(filepath, params, logger):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        if not formatted:
            df = format_df(df, params, resp_mapping, color_correct_mapping)
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

def format_df(df:pd.DataFrame, params:dict, resp_mapping:dict, color_correct_mapping:dict) -> pd.DataFrame:
    """Reshape a raw Stroop CSV into the library's standard column layout.

    Steps:
      1. Mask to trial rows via `cols.trial` non-null.
      2. Copy each configured metacol and `cols` entry, warning when a referenced
         CSV column is missing.
      3. Parse list-valued response/rt cells into Python lists.
      4. Map response cells (keyboard letter OR touch shape name) to a
         standardized option int via `resp_mapping`.
      5. Derive `correct_opt` from `this_color` via `color_correct_mapping`,
         and `correct = (response_first == correct_opt)`.
    """
    fmtdf = pd.DataFrame()
    mask = np.invert(df[params['cols']['trial']].isna())

    # PsychoPy records `blocks.thisN` only on the end-of-block marker rows
    # (one row per block), not on the trial rows themselves. Backfill so each
    # trial row carries the index of the block it belongs to. Done on a shallow
    # copy so we don't mutate the caller's dataframe.
    block_csv_col = params['cols'].get('block')
    if block_csv_col and block_csv_col in df.columns:
        if df.loc[mask, block_csv_col].isna().all() and df[block_csv_col].notna().any():
            df = df.copy()
            df[block_csv_col] = df[block_csv_col].bfill()

    copy_configured_columns(fmtdf, df, params['metacols'], 'metacols', mask=mask, logger=logger)
    copy_configured_columns(fmtdf, df, params['cols'], 'cols', mask=mask, logger=logger)

    # Parse list-string cells like `'["l","j"]'` into actual Python lists.
    # We pass slice(None) to keep the entire list — multiple-response trials need
    # both first and last for later inspection, and rt list is parsed the same way.
    for resp_col in ('response', 'rt'):
        if resp_col in fmtdf.columns:
            fmtdf[resp_col] = fmtdf[resp_col].apply(
                lambda x: handle_multiple_responses(x, slice_index=slice(None))
            )

    fmtdf = parse_responses(fmtdf, resp_mapping, color_correct_mapping)
    return fmtdf


def _normalize_resp_list(value, resp_mapping:dict) -> list:
    """Coerce a response cell into a list of option ints (or pass-through values).

    Cells reach this function as one of:
      - Python list (parsed earlier by `handle_multiple_responses`),
      - scalar string ("j", or rarely an already-int-like value),
      - None / NaN.
    Each element is mapped through `resp_mapping` when the key is recognised,
    otherwise passed through (so an unexpected token shows up as itself rather
    than being silently dropped).
    """
    if isinstance(value, list):
        return [resp_mapping.get(r, r) for r in value]
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    return [resp_mapping.get(value, value)]


def _normalize_rt_list(value) -> list:
    """Coerce an RT cell into a list of floats."""
    if isinstance(value, list):
        out:list = []
        for r in value:
            try:
                out.append(float(r))
            except (TypeError, ValueError):
                out.append(np.nan)
        return out
    if value is None:
        return []
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def parse_responses(df:pd.DataFrame, resp_mapping:dict, color_correct_mapping:dict) -> pd.DataFrame:
    """Compute per-trial derived response columns.

    Writes (when the underlying columns are available):

      - `num_responses`      — count of response tokens recorded on the trial.
      - `response_first`     — first response, mapped to an option int.
      - `response_last`      — last response, mapped to an option int.
      - `rt_first`           — first RT (the "stimulus-driven" reaction time,
        which is what's used for scoring; standard Stroop convention).
      - `rt_last`            — last RT (kept for downstream inspection of
        self-corrections on multi-response trials).
      - `correct_opt`        — option int the participant *should* have selected
        based on `this_color` × `color_correct_mapping`.
      - `correct`            — 1 if `response_first == correct_opt`, else 0,
        NaN if either side could not be derived.
    """
    df = df.copy()
    if 'response' in df.columns:
        resp_lists = df['response'].apply(lambda v: _normalize_resp_list(v, resp_mapping))
    else:
        resp_lists = pd.Series([[]] * len(df), index=df.index)
    if 'rt' in df.columns:
        rt_lists = df['rt'].apply(_normalize_rt_list)
    else:
        rt_lists = pd.Series([[]] * len(df), index=df.index)

    def _first(l): return l[0] if l else np.nan
    def _last(l): return l[-1] if l else np.nan

    df['num_responses'] = resp_lists.apply(len).astype(float)
    df['response_first'] = resp_lists.apply(_first)
    df['response_last'] = resp_lists.apply(_last)
    df['rt_first'] = rt_lists.apply(_first)
    df['rt_last'] = rt_lists.apply(_last)

    # Derive correct_opt from this_color via the config-supplied mapping. The
    # CSVs in this experiment don't ship an explicit correct-response column for
    # real trials, so this is the only path to a correctness signal.
    if 'this_color' in df.columns and color_correct_mapping:
        df['correct_opt'] = df['this_color'].map(color_correct_mapping)
    elif 'correct_resp' in df.columns:
        # Future variants might have an explicit column; respect it if so.
        df['correct_opt'] = df['correct_resp']

    if 'correct_opt' in df.columns:
        # Correct only when both sides are non-NaN AND match. NaN propagation via
        # comparison would give False for NaN==NaN, but we want NaN to mean
        # "couldn't determine" rather than "incorrect".
        both_present = df['response_first'].notna() & df['correct_opt'].notna()
        df['correct'] = np.where(
            both_present,
            (df['response_first'] == df['correct_opt']).astype(float),
            np.nan,
        )
    return df


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_df(df:pd.DataFrame, trial_filter:str) -> pd.DataFrame:
    """Build a 1-row dataframe of per-file Stroop scores.

    Groups by `test` (classic / emotional) × `condition`
    (congruent / incongruent / neutral / negative / positive). For each bucket:

      `<test>_<condition>_n_trials`
      `<test>_<condition>_prop_correct`
      `<test>_<condition>_mean_rt_correct`
      `<test>_<condition>_sd_rt_correct`
      `<test>_<condition>_mean_rt_incorrect`
      `<test>_<condition>_sd_rt_incorrect`

    Plus three derived Stroop-interference scores (mean correct RT differences),
    when the underlying buckets are populated:

      `classic_stroop_interference_rt`           — incongruent − congruent
      `emotional_negative_interference_rt`       — negative   − neutral
      `emotional_positive_interference_rt`       — positive   − neutral

    :param df: per-trial dataframe (produced by `format_df`).
    :param trial_filter: optional pandas query string applied before scoring.
    """
    if trial_filter:
        df = df.query(trial_filter)

    score_dict:dict = {}
    if 'test' not in df.columns or 'condition' not in df.columns:
        return pd.DataFrame(score_dict, index=[0])

    grouped = df.groupby(['test', 'condition'], dropna=False)
    for (test, cond), bucket in grouped:
        if not isinstance(test, str) or not isinstance(cond, str):
            continue
        prefix = f'{test}_{cond}'
        score_dict[f'{prefix}_n_trials'] = len(bucket)
        if 'correct' in bucket.columns:
            score_dict[f'{prefix}_prop_correct'] = bucket['correct'].mean()
            correct_rt = bucket.loc[bucket['correct'] == 1, 'rt_first'] if 'rt_first' in bucket.columns else pd.Series(dtype=float)
            incorrect_rt = bucket.loc[bucket['correct'] == 0, 'rt_first'] if 'rt_first' in bucket.columns else pd.Series(dtype=float)
            score_dict[f'{prefix}_mean_rt_correct'] = correct_rt.mean()
            score_dict[f'{prefix}_sd_rt_correct'] = correct_rt.std()
            score_dict[f'{prefix}_mean_rt_incorrect'] = incorrect_rt.mean()
            score_dict[f'{prefix}_sd_rt_incorrect'] = incorrect_rt.std()

    # Derived Stroop-interference contrasts. The `_diff` helper returns NaN
    # when either operand is missing, so the column is always written.
    score_dict['classic_stroop_interference_rt'] = _diff(
        score_dict, 'classic_incongruent_mean_rt_correct', 'classic_congruent_mean_rt_correct')
    score_dict['emotional_negative_interference_rt'] = _diff(
        score_dict, 'emotional_negative_mean_rt_correct', 'emotional_neutral_mean_rt_correct')
    score_dict['emotional_positive_interference_rt'] = _diff(
        score_dict, 'emotional_positive_mean_rt_correct', 'emotional_neutral_mean_rt_correct')

    return pd.DataFrame(score_dict, index=[0])


def _diff(d:dict, a_key:str, b_key:str):
    """Return `d[a_key] - d[b_key]`, or NaN if either is missing/NaN."""
    a = d.get(a_key)
    b = d.get(b_key)
    if a is None or b is None:
        return np.nan
    try:
        if np.isnan(a) or np.isnan(b):
            return np.nan
    except TypeError:
        return np.nan
    return a - b
