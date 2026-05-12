'''
Synonyms task scoring.

Reads PsychoPy CSV output of a synonym-matching task, maps keyboard / touch
responses to numeric option indices via a configurable mapping, and computes
per-participant accuracy and RT summary statistics.
'''

import os
import logging
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

REQUIRED_PARAMS = {
    'metacols': dict,
    'cols': dict,
}

# Fallback response mapping used only when the JSON config does not supply its own.
# Prefer setting `"resp_mapping": {...}` at the top level of the synonyms config.
DEFAULT_RESP_MAPPING = {
    "n":1,
    "opt1_shape":1,
    "m":2,
    "opt2_shape":2,
    ",":3,
    "opt3_shape":3,
    "comma":3,
    ".":4,
    "period":4,
    "opt4_shape":4
}


def _get_resp_mapping(params: dict) -> dict:
    """Return the response-key → option-index mapping for this run.

    Prefers `params['resp_mapping']` if the config supplies it (per-experiment
    custom keys); otherwise falls back to `DEFAULT_RESP_MAPPING` and logs a
    warning so future configs are encouraged to be explicit.
    """
    mapping = params.get('resp_mapping')
    if mapping is None:
        logging.getLogger('root').warning(
            "synonyms: no 'resp_mapping' in params; falling back to DEFAULT_RESP_MAPPING. "
            "Adding 'resp_mapping' to your JSON config is recommended so the mapping is explicit per experiment."
        )
        return DEFAULT_RESP_MAPPING
    return mapping


def synonyms(params:dict, out:str=os.getcwd(), write:bool=True, filelist:str|list='', formatted:bool=False, log=20,
             trial_filter:str='') -> tuple:
    """Score one or more Synonyms data files.

    :param params: configuration dict (see `tests/synonyms_example.json`). May include `'resp_mapping'`
        to override the built-in keyboard/touch mapping.
    :param out: output directory.
    :param write: if True, write combined trials + scores CSVs.
    :param filelist: list of CSV paths, path to a text file with one CSV per line, or empty for GUI picker.
    :param formatted: True if the input is already tidy with standard column names.
    :param log: log level.
    :param trial_filter: optional pandas query string to subset trials before scoring.
    :returns: (combined_scores, combined_trials).
    """
    setup_logger(name='root', out=out, level=log).info('start')
    validate_params(params, REQUIRED_PARAMS)
    resp_mapping = _get_resp_mapping(params)

    def process_one(filepath, params, logger):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        if not formatted:
            df = format_df(df, params, resp_mapping)
        df = parse_responses(df, resp_mapping)
        df.insert(1, 'filename', filename)
        scores_row = pd.concat([get_meta_cols(df, params), score_df(df, trial_filter)], axis=1)
        scores_row.insert(1, 'filename', filename)
        return df, scores_row

    return run_task(
        params=params, filelist=filelist, out=out, write=write,
        process_file_fn=process_one,
    )


def format_df(df:pd.DataFrame, params:dict, resp_mapping:dict=DEFAULT_RESP_MAPPING) -> pd.DataFrame:
    """Reshape a raw Synonyms CSV into the library's standard column layout.

    Masks to trial rows, renames per the JSON config, parses list-valued
    `response`/`rt` cells, then maps the human-readable response cells
    (keyboard key like `'n'`, or touch label like `'opt1_shape'`) to their
    integer option index (1-4) via `resp_mapping`. Both `response` and
    `correct_resp` are normalised through the same mapping so a later
    `correct_resp in response` comparison works regardless of input form.
    """
    fmtdf = pd.DataFrame()
    mask = np.invert(df[params['cols']['trial']].isna())

    # Copy each metacol & each trial-level col. `copy_configured_columns`
    # warns when a configured column is missing from the CSV.
    copy_configured_columns(fmtdf, df, params['metacols'], 'metacols', mask=mask)
    copy_configured_columns(fmtdf, df, params['cols'], 'cols', mask=mask)

    for resp_col in ['response', 'rt']:
        if resp_col in fmtdf.columns:
            fmtdf[resp_col] = fmtdf[resp_col].apply(lambda x: handle_multiple_responses(x, slice_index=slice(None)))

    for opt_col in ['response', 'correct_resp']:
        if opt_col in fmtdf.columns:
            fmtdf[opt_col] = fmtdf[opt_col].apply(
                lambda x: [resp_mapping.get(resp, resp) for resp in x] if isinstance(x, list)
                else resp_mapping.get(x, x)
            )

    return fmtdf


def _normalize_resp_list(value, resp_mapping: dict) -> list:
    """Coerce a response cell into a list of ints (or mapped values).

    Cells reach this function as either a Python list (when the CSV cell was a
    list-string that handle_multiple_responses parsed) or as a scalar (string
    keyboard key, or already-mapped int). None becomes [].
    A NaN scalar gets wrapped as [NaN] to preserve the historical num_responses
    count of 1 for non-response trials.
    """
    if isinstance(value, list):
        return [int(r) if str(r).isdigit() else resp_mapping.get(r, r) for r in value]
    if value is None:
        return []
    if str(value).isdigit():
        return [int(value)]
    return [resp_mapping.get(value, value)]


def _normalize_rt_list(value) -> list:
    """Coerce an RT cell into a list of floats (mirror of _normalize_resp_list)."""
    if isinstance(value, list):
        return [float(r) for r in value]
    if value is None:
        return []
    return [float(value)]  # float(nan) → nan; preserves historical behavior


def parse_responses(df:pd.DataFrame, resp_mapping:dict=DEFAULT_RESP_MAPPING):
    """Compute per-trial derived response columns from `response`, `rt`, `correct_resp`.

    Writes: `num_responses`, `response_last`, `rt_last`, `correct`, `correct_resp_index`.
    No in-place mutation of df cells — derived columns are computed via `.apply` and
    appended in one assignment so SettingWithCopyWarning is impossible.
    """
    df = df.copy()
    if 'response' not in df.columns or 'rt' not in df.columns:
        return df

    resp_lists = df['response'].apply(lambda v: _normalize_resp_list(v, resp_mapping))
    rt_lists = df['rt'].apply(_normalize_rt_list)

    def _correct_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    correct_resp_int = df['correct_resp'].apply(_correct_int) if 'correct_resp' in df.columns else pd.Series([None] * len(df), index=df.index)

    n = resp_lists.apply(len)
    response_last = resp_lists.apply(lambda l: l[-1] if l else np.nan)
    rt_last = rt_lists.apply(lambda l: l[-1] if l else np.nan)

    def _idx(row_resp, target):
        if not row_resp or target is None:
            return np.nan
        return float(row_resp.index(target)) if target in row_resp else np.nan
    correct_resp_index = pd.Series(
        [_idx(r, t) for r, t in zip(resp_lists, correct_resp_int)],
        index=df.index,
        dtype='float64',
    )
    correct = (~correct_resp_index.isna()).astype(float)
    # When there are zero responses, the original code sets correct=0 explicitly — that
    # falls out naturally above since correct_resp_index stays NaN.

    df['num_responses'] = n.astype(float)
    df['response_last'] = response_last
    df['rt_last'] = rt_last
    df['correct'] = correct
    df['correct_resp_index'] = correct_resp_index
    return df


def score_df(df:pd.DataFrame, trial_filter:str) -> pd.DataFrame:
    """Build a 1-row dataframe of per-file Synonyms scores.

    Columns: num_correct, prop_correct, mean_rt, sd_rt, plus the same RT
    summaries restricted to correct trials and to incorrect trials.

    :param trial_filter: optional pandas query string applied before scoring.
    """
    score_dict = {}

    if trial_filter:
        df = df.query(trial_filter)

    score_dict['num_correct'] = df['correct'].sum()
    score_dict['prop_correct'] = df['correct'].mean()
    score_dict['mean_rt'] = df['rt_last'].mean()
    score_dict['sd_rt'] = df['rt_last'].std()
    score_dict['mean_correct_resp_rt'] = df.loc[df['correct']==1, 'rt_last'].mean()
    score_dict['std_correct_resp_rt'] = df.loc[df['correct']==1, 'rt_last'].std()
    score_dict['mean_incorrect_resp_rt'] = df.loc[df['correct']==0, 'rt_last'].mean()
    score_dict['std_incorrect_resp_rt'] = df.loc[df['correct']==0, 'rt_last'].std()

    return pd.DataFrame(score_dict, index=[0])
