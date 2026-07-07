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
    get_meta_cols,
    handle_multiple_responses,
    validate_params,
    run_task,
    copy_configured_columns,
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


def sert(params:dict, out:str=os.getcwd(), write:bool=True, filelist:str|list='', formatted:bool=False, log=20, logfile:bool=False,
         trial_filter:str='', block_switch_rep:bool=False) -> tuple:
    """Score one or more SERT data files.

    Trial-level switch/repeat scoring (each trial vs the previous trial's cue
    within its block) is always produced under ``<event_type>_trial_*`` columns.
    Block-level switch/repeat scoring (whole-block switch vs repeat, the
    PsychoPy/Pavlovia design) is opt-in via ``block_switch_rep`` and emits the
    ``<event_type>_switch/repeat/switch_cost_*`` columns. It's opt-in because
    sources like MetricWire don't run fixed switch/repeat blocks.

    :param params: configuration dict (see `tests/sert_example.json` or `sert_example_touch.json`).
    :param out: output directory.
    :param write: if True, write combined trials + scores CSVs.
    :param filelist: list of CSV paths, path to a text file with one CSV per line, or empty for GUI picker.
    :param formatted: True if the input is already tidy with standard column names.
    :param log: log level.
    :param logfile: if True, write a timestamped ``log_<ts>.log`` to ``out`` (default False).
    :param trial_filter: optional pandas query string to subset trials before scoring.
    :param block_switch_rep: if True, also compute block-level switch/repeat scores
        (requires fixed switch/repeat blocks, as in the Pavlovia SERT).
    :returns: (combined_scores, combined_trials).
    """
    setup_logger(out=out, level=log, logfile=logfile).info('start')
    validate_params(params, REQUIRED_PARAMS)

    def process_one(filepath, params, logger):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        if not formatted:
            df = format_df(df, params)
        df = parse_choice_columns(df)
        df = add_blocks(df)
        df = add_trial_switch_rep(df)
        if block_switch_rep:
            df = add_block_switch_rep(df)
        df = parse_responses(df)
        df.insert(1, 'filename', filename)
        scores_row = pd.concat(
            [get_meta_cols(df, params), score_df(df, trial_filter, block_switch_rep)], axis=1)
        scores_row.insert(1, 'filename', filename)
        return df, scores_row

    return run_task(
        params=params, filelist=filelist, out=out, write=write,
        process_file_fn=process_one,
    )


def format_df(df:pd.DataFrame, params:dict) -> pd.DataFrame:
    """Reshape a raw PsychoPy SERT CSV into the library's standard column layout.

    Drops non-trial rows by masking on `params['cols']['trial']` (rows where
    trial number is NaN are skipped — that's how PsychoPy marks header/intro/
    practice rows). Renames the experiment's columns to standard names per
    the JSON config, parses list-valued `response`/`rt` cells back into Python
    lists, and maps touchscreen labels like `'LeftImage'` to integer response
    codes 1/2/3.
    """
    fmtdf = pd.DataFrame()
    # Trial-counter NaN marks non-trial rows; invert to keep only real trials.
    mask = np.invert(df[params['cols']['trial']].isna())

    # Copy each metacol & each trial-level col that the config names and the CSV
    # actually has. `copy_configured_columns` warns when a configured column
    # name is missing from the CSV — typo-debugging aid.
    copy_configured_columns(fmtdf, df, params['metacols'], 'metacols', mask=mask)
    copy_configured_columns(fmtdf, df, params['cols'], 'cols', mask=mask)

    # Normalize cue labels that vary across task versions. Some CSVs record the
    # lethality cue as 'lethal'; standardise to 'lethality' so score column names
    # and switch-cost contrasts are consistent across all participants.
    if 'cue' in fmtdf.columns:
        fmtdf['cue'] = fmtdf['cue'].replace({'lethal': 'lethality'})

    # Cells like "['n']" become Python lists; scalar cells pass through unchanged.
    for resp_col in ['response', 'rt']:
        if resp_col in fmtdf.columns:
            fmtdf[resp_col] = fmtdf[resp_col].apply(lambda x: handle_multiple_responses(x, slice_index=slice(None)))

    # Map touch labels (LeftImage / MiddleImage / RightImage) to 1/2/3 so keyboard
    # and touch responses share the same integer namespace downstream.
    fmtdf['response'] = fmtdf['response'].apply(
        lambda x: [touch_resp_mapping.get(resp, resp) for resp in x] if isinstance(x, list)
        else touch_resp_mapping.get(x, x)
    )
    return fmtdf


def add_blocks(df:pd.DataFrame) -> pd.DataFrame:
    """Ensure a `block` column, used to group trials for switch/repeat scoring.

    Uses an existing `block` column when the config maps one (via `cols.block`) —
    needed when the source already carries a block index or uses a per-block
    trial counter that resets (e.g. MetricWire). Otherwise derives blocks from a
    continuous trial counter as fixed 10-trial runs (`trial // 10`), the
    PsychoPy/Pavlovia default.
    """
    df = df.copy()
    if 'block' not in df.columns:
        # Integer-divide the continuous trial number by 10 (1-indexed) to derive block.
        df['block'] = ((df['trial']) // 10) + 1
    return df


def add_trial_switch_rep(df:pd.DataFrame) -> pd.DataFrame:
    """Add trial-level `trial_switch_rep`: cue vs the previous trial's cue.

    Within each block (in row order, which is trial order), a trial is `'switch'`
    when its `cue` differs from the immediately preceding trial's, `'repeat'` when
    it matches, and `'first'` for the first trial of a block (no predecessor;
    excluded from switch-cost contrasts). This is the meaningful contrast for
    designs that switch cues within a run rather than between fixed blocks.
    """
    df = df.copy()
    prev_cue = df.groupby('block')['cue'].shift()
    df['trial_switch_rep'] = np.where(
        prev_cue.isna(), 'first',
        np.where(df['cue'] != prev_cue, 'switch', 'repeat'))
    return df


def add_block_switch_rep(df:pd.DataFrame) -> pd.DataFrame:
    """Add block-level `block_nunique_cues` and `block_switch_rep` columns.

    A block with more than one distinct `cue` value contains a cue switch and is
    labelled `'switch'`; one with a single cue throughout is `'repeat'`. This is
    the fixed-block design used by the Pavlovia SERT.
    """
    df = df.copy()
    # `transform('nunique')` broadcasts the per-block unique cue count back onto every row.
    df['block_nunique_cues'] = df.groupby('block')['cue'].transform('nunique')
    df['block_switch_rep'] = np.where(df['block_nunique_cues'] > 1, 'switch', 'repeat')
    return df


def parse_choice_value(value:str) -> dict:
    """Parse a SERT choice-stimulus filename into its semantic components.

    Stimulus filenames look like `safe_apple_orange_oval.png` — class /
    type tokens / color / shape, separated by spaces, underscores, or hyphens.
    This function picks them apart, normalising `rectangle` → `rect` and
    treating unknown shapes/colors as absent. Returns a dict with the four
    keys `class`, `type`, `color`, `shape`; NaN input maps to all-None.

    Example: `'lethal_red_apple_orange_oval.png'` →
    `{'class': 'lethal', 'type': 'red_apple', 'color': 'orange', 'shape': 'oval'}`.
    """
    if pd.isna(value):
        return {'class': None, 'type': None, 'color': None, 'shape': None}

    # Drop the file extension and split on any combination of ` ` `_` `-`.
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
    """For each of `left_choice` / `middle_choice` / `right_choice`, expand into 4 derived cols.

    Each `<side>_choice` column holds a stimulus filename; this function calls
    `parse_choice_value` on each cell and inserts `<side>_choice_class`,
    `<side>_choice_type`, `<side>_choice_color`, `<side>_choice_shape` columns
    next to the original. Sides whose source column isn't present are skipped.
    """
    for side in ['left', 'middle', 'right']:
        choice_col = f'{side}_choice'
        if choice_col not in df.columns:
            continue
        # `apply` returns a Series of dicts; second `apply(pd.Series)` expands those dicts to columns.
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
        # Defensive: rt_list and resp_list normally have the same length, but bail to
        # NaN if a participant's CSV row is malformed (e.g. responses without RTs).
        if pd.isna(idx) or not rt_list or int(idx) >= len(rt_list):
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
    """Store `sdict[a_key] - sdict[b_key]` at `out_key`, or NaN if either is missing.

    Used to compute switch-cost differences when one of the input metrics
    may not have been produced (e.g. a participant had no `repeat` trials of
    a given cue, so `{prefix}_repeat_{metric}` was never written).
    """
    if a_key in sdict and b_key in sdict:
        sdict[out_key] = sdict[a_key] - sdict[b_key]
    else:
        sdict[out_key] = np.nan


def _bucket_metrics(score_dict: dict, prefix: str, group: pd.DataFrame) -> None:
    """Write the standard bucket metric set into score_dict under `prefix`.

    Computes nine numbers: trial count, correct count, accuracy, mean/median/SD
    of first-response RT, and mean/median/SD of correct-response RT. Used at
    each level of the (event_type × switch_rep × cue) score grid so every
    bucket emits the same columns under its own `<event_type>_<switch_rep>_<cue>_…` prefix.
    """
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


def _score_switch_rep(score_dict:dict, event_type, event_group:pd.DataFrame,
                      sr_col:str, tag:str) -> None:
    """Write the switch/repeat × cue bucket metrics + switch-cost diffs.

    `sr_col` is the switch/repeat column to group on (`trial_switch_rep` or
    `block_switch_rep`). `tag` is inserted into the column prefix so trial- and
    block-level results don't collide: block-level uses ``tag=''`` (yielding the
    original ``<event_type>_switch/repeat/switch_cost_*`` names) and trial-level
    uses ``tag='trial_'``. Switch cost is computed for the `switch` and `repeat`
    labels only (a trial-level `first` group, if present, is ignored here).
    """
    base = f'{event_type}_{tag}'
    for switch_rep, switch_rep_group in event_group.groupby(sr_col):
        # The trial-level 'first' label (first trial of a block, no predecessor)
        # is neither switch nor repeat — exclude it from the switch/repeat grid.
        if switch_rep == 'first':
            continue
        _bucket_metrics(score_dict, f'{base}{switch_rep}', switch_rep_group)
        for cue, cue_group in switch_rep_group.groupby('cue'):
            _bucket_metrics(score_dict, f'{base}{switch_rep}_{cue}', cue_group)

    # Switch-cost differences: switch minus repeat, for each (cue, metric).
    for cue_label in _SWITCH_COST_CUES:
        cue_suffix = f'_{cue_label}' if cue_label else ''
        for metric in _SWITCH_COST_METRICS:
            safe_diff(
                score_dict,
                f'{base}switch_cost{cue_suffix}_{metric}',
                f'{base}switch{cue_suffix}_{metric}',
                f'{base}repeat{cue_suffix}_{metric}',
            )


def score_df(df:pd.DataFrame, trial_filter:str, block_switch_rep:bool=False) -> pd.DataFrame:
    """Build a 1-row dataframe of per-file SERT scores.

    For each `event_type`, writes overall bucket metrics, then the trial-level
    switch/repeat × cue grid (always, under ``<event_type>_trial_*``) and, when
    `block_switch_rep` is True, the block-level grid (under the original
    ``<event_type>_switch/repeat/switch_cost_*`` names). `_bucket_metrics` is
    called at every grain; switch-cost differences are `switch - repeat`.

    :param df: per-trial dataframe (populated by parse_responses, add_blocks,
        add_trial_switch_rep, and — when block-level is requested — add_block_switch_rep).
    :param trial_filter: optional pandas query string applied before scoring.
    :param block_switch_rep: if True, also emit the block-level switch/repeat grid.
    """
    score_dict = {}

    if trial_filter:
        df = df.query(trial_filter)

    for event_type, event_group in df.groupby('event_type'):
        _bucket_metrics(score_dict, str(event_type), event_group)
        _score_switch_rep(score_dict, event_type, event_group, 'trial_switch_rep', 'trial_')
        if block_switch_rep:
            _score_switch_rep(score_dict, event_type, event_group, 'block_switch_rep', '')

    return pd.DataFrame(score_dict, index=[0])
