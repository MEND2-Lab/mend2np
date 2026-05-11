'''
SERT (Suicide Emotion Rigidity Task) scoring.

Reads PsychoPy CSV output of the SERT, parses the per-trial choice stimuli into
their semantic components (class / type / color / shape), classifies trials as
switch vs. repeat blocks, scores accuracy and RTs by event_type × switch/repeat
× cue dimension, and computes switch-cost difference scores for each
(metric, cue) combination.
'''

import re
import os
from pathlib import Path
import pandas as pd
import numpy as np
from mend2np.utils import (
    setup_logger,
    write_out,
    get_meta_cols,
    handle_multiple_responses,
    validate_params,
    run_task,
)

touch_resp_mapping = {
    'LeftImage':1,
    'MiddleImage':2,
    'RightImage':3
}

REQUIRED_PARAMS = {
    'metacols': dict,
    'cols': dict,
}

# Per-bucket score metrics — same set is computed at every level of the
# (event_type) × (switch/repeat) × (cue) grid.
_BUCKET_METRICS = (
    'num_trials',
    'num_correct',
    'accuracy',
    'mean_first_rt',
    'median_first_rt',
    'std_first_rt',
    'mean_correct_resp_rt',
    'median_correct_resp_rt',
    'std_correct_resp_rt',
)

# Cue dimensions over which switch-cost differences are reported. '' = the
# bucket-level switch cost (no cue split). The named entries are the cue values
# the original code hard-coded.
_SWITCH_COST_CUES = ('', 'color', 'shape', 'lethality')

# Subset of bucket metrics that switch-cost differences are computed for.
_SWITCH_COST_METRICS = (
    'mean_first_rt',
    'median_first_rt',
    'mean_correct_resp_rt',
    'median_correct_resp_rt',
    'accuracy',
    'num_correct',
)


def sert(params:dict, out:str=os.getcwd(), write:bool=True, filelist:str|list='', formatted:bool=False, log=20,
         trial_filter:str='') -> tuple:
    """Score one or more SERT data files.

    :param params: configuration dict (see `tests/sert_example.json` or `sert_example_touch.json`).
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

    def process_one(filepath, params, logger):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        if not formatted:
            df = format_df(df, params)
        df = parse_choice_columns(df)
        df = add_switch_rep(df)
        df = parse_responses(df)
        df.insert(1, 'filename', filename)
        scores_row = pd.concat([get_meta_cols(df, params), score_df(df, trial_filter)], axis=1)
        scores_row.insert(1, 'filename', filename)
        return df, scores_row

    return run_task(
        params=params, filelist=filelist, out=out, write=write, log=log,
        process_file_fn=process_one,
    )


def format_df(df:pd.DataFrame, params:dict) -> pd.DataFrame:
    fmtdf = pd.DataFrame()
    mask = np.invert(df[params['cols']['trial']].isna())

    for metacol in params['metacols']:
        if params['metacols'][metacol] and params['metacols'][metacol] in df.columns:
            fmtdf[metacol] = df.loc[mask, params['metacols'][metacol]]

    for col in params['cols']:
        if params['cols'][col] and params['cols'][col] in df.columns:
            fmtdf[col] = df.loc[mask, params['cols'][col]]

    for resp_col in ['response', 'rt']:
        if resp_col in fmtdf.columns:
            fmtdf[resp_col] = fmtdf[resp_col].apply(lambda x: handle_multiple_responses(x, slice_index=slice(None)))

    fmtdf['response'] = fmtdf['response'].apply(
        lambda x: [touch_resp_mapping.get(resp, resp) for resp in x] if isinstance(x, list)
        else touch_resp_mapping.get(x, x)
    )
    return fmtdf


def add_switch_rep(df:pd.DataFrame) -> pd.DataFrame:
    df['block'] = ((df['trial']) // 10) + 1
    df['block_nunique_cues'] = df.groupby('block')['cue'].transform('nunique')
    df['block_switch_rep'] = np.where(df['block_nunique_cues'] > 1, 'switch', 'repeat')
    return df


def parse_choice_value(value:str) -> dict:
    if pd.isna(value):
        return {'class': None, 'type': None, 'color': None, 'shape': None}

    raw = str(Path(value).stem)
    tokens = re.split(r'[ _\-]+', raw)
    lower_tokens = [t.lower() for t in tokens]

    obj_class = tokens[0] if lower_tokens and lower_tokens[0] in ['safe', 'inert', 'lethal'] else None

    shape = None
    color = None
    if lower_tokens:
        if lower_tokens[-1] in ['oval', 'rhom', 'rect', 'rectangle']:
            shape = tokens[-1]
        if len(tokens) >= 2 and lower_tokens[-2] in ['orange', 'blue']:
            color = tokens[-2]

    start = 1 if obj_class else 0
    end = len(tokens)
    if shape is not None:
        end -= 1
    if color is not None:
        end -= 1

    type_tokens = tokens[start:end]
    obj_type = '_'.join(type_tokens) if type_tokens else None

    return {
        'class': obj_class,
        'type': obj_type,
        'color': color,
        'shape': 'rect' if shape == 'rectangle' else shape,
    }


def parse_choice_columns(df:pd.DataFrame) -> pd.DataFrame:
    for side in ['left', 'middle', 'right']:
        choice_col = f'{side}_choice'
        if choice_col not in df.columns:
            continue
        parsed = df[choice_col].apply(parse_choice_value).apply(pd.Series)
        parsed.columns = [f'{choice_col}_{suffix}' for suffix in parsed.columns]
        df = pd.concat([df, parsed], axis=1)
    return df


def _normalize_response(value):
    """Coerce a sert response cell into a Python list of mapped values.

    Mirrors the original iterrows-based logic. Note the quirky NaN handling: a
    NaN cell falls through to the `isalpha` branch (because `str(float('nan'))`
    is `"nan"`, which is alpha) and ends up as `[nan]`. That quirk is preserved
    so per-trial outputs stay byte-identical with the historical scores.
    """
    if isinstance(value, list):
        return [int(r) if str(r).isdigit() else touch_resp_mapping.get(r, r) for r in value]
    if value is None:
        return []
    s = str(value)
    if s.isdigit():
        return [int(value)]
    if s.isalpha():  # includes NaN ("nan")
        return [touch_resp_mapping.get(value, value)]
    return [value]


def _normalize_rt(value):
    if isinstance(value, list):
        return [float(r) for r in value]
    if value is None:
        return []
    return [float(value)]  # float(nan) is still nan — preserves the NaN value


def parse_responses(df:pd.DataFrame) -> pd.DataFrame:
    """Compute derived response columns: num_responses, first/last response + RT,
    correct flag, correct_resp_index, correct_resp_rt. No in-place mutation."""
    df = df.copy()
    if 'response' not in df.columns or 'rt' not in df.columns:
        return df

    resp_lists = df['response'].apply(_normalize_response)
    rt_lists = df['rt'].apply(_normalize_rt)

    def _to_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    correct_resp_int = (df['correct_resp'].apply(_to_int)
                       if 'correct_resp' in df.columns
                       else pd.Series([None] * len(df), index=df.index))

    n = resp_lists.apply(len)

    def _first(l): return l[0] if l else np.nan
    def _last(l): return l[-1] if l else np.nan

    first_response = resp_lists.apply(_first)
    last_response = resp_lists.apply(_last)
    first_response_rt = rt_lists.apply(_first)
    last_response_rt = rt_lists.apply(_last)

    def _correct_index(resp_list, target):
        if not resp_list or target is None or target not in resp_list:
            return np.nan
        return float(resp_list.index(target))
    correct_resp_index = pd.Series(
        [_correct_index(r, t) for r, t in zip(resp_lists, correct_resp_int)],
        index=df.index, dtype='float64',
    )

    def _correct_rt(rt_list, idx):
        if pd.isna(idx) or not rt_list:
            return np.nan
        return rt_list[int(idx)]
    correct_resp_rt = pd.Series(
        [_correct_rt(r, i) for r, i in zip(rt_lists, correct_resp_index)],
        index=df.index, dtype='float64',
    )

    correct = (~correct_resp_index.isna()).astype(float)

    df['num_responses'] = n.astype(float)
    df['first_response'] = first_response
    df['first_response_rt'] = first_response_rt
    df['last_response'] = last_response
    df['last_response_rt'] = last_response_rt
    df['correct'] = correct
    df['correct_resp_index'] = correct_resp_index
    df['correct_resp_rt'] = correct_resp_rt
    return df


def safe_diff(sdict: dict, out_key: str, a_key: str, b_key: str):
    if a_key in sdict and b_key in sdict:
        sdict[out_key] = sdict[a_key] - sdict[b_key]
    else:
        sdict[out_key] = np.nan


def _bucket_metrics(score_dict: dict, prefix: str, group: pd.DataFrame) -> None:
    """Write the standard bucket metric set into score_dict under `prefix`."""
    correct_only = group.loc[group['correct'] == 1, 'correct_resp_rt']
    score_dict[f'{prefix}_num_trials'] = len(group)
    score_dict[f'{prefix}_num_correct'] = group['correct'].sum()
    score_dict[f'{prefix}_accuracy'] = group['correct'].mean()
    score_dict[f'{prefix}_mean_first_rt'] = group['first_response_rt'].mean()
    score_dict[f'{prefix}_median_first_rt'] = group['first_response_rt'].median()
    score_dict[f'{prefix}_std_first_rt'] = group['first_response_rt'].std()
    score_dict[f'{prefix}_mean_correct_resp_rt'] = correct_only.mean()
    score_dict[f'{prefix}_median_correct_resp_rt'] = correct_only.median()
    score_dict[f'{prefix}_std_correct_resp_rt'] = correct_only.std()


def score_df(df:pd.DataFrame, trial_filter:str) -> pd.DataFrame:
    score_dict = {}

    if trial_filter:
        df = df.query(trial_filter)

    for event_type, event_group in df.groupby('event_type'):

        _bucket_metrics(score_dict, str(event_type), event_group)

        for switch_rep, switch_rep_group in event_group.groupby('block_switch_rep'):
            _bucket_metrics(score_dict, f'{event_type}_{switch_rep}', switch_rep_group)

            for cue, cue_group in switch_rep_group.groupby('cue'):
                _bucket_metrics(score_dict, f'{event_type}_{switch_rep}_{cue}', cue_group)

        # Switch-cost differences: switch minus repeat, for each (cue, metric) combination.
        for cue_label in _SWITCH_COST_CUES:
            cue_suffix = f'_{cue_label}' if cue_label else ''
            for metric in _SWITCH_COST_METRICS:
                safe_diff(
                    score_dict,
                    f'{event_type}_switch_cost{cue_suffix}_{metric}',
                    f'{event_type}_switch{cue_suffix}_{metric}',
                    f'{event_type}_repeat{cue_suffix}_{metric}',
                )

    return pd.DataFrame(score_dict, index=[0])
