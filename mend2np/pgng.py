'''
PGNG (Parametric Go / No-go / Stop) scoring.

Reads PsychoPy or E-Prime CSV output from the PGNG task, classifies trials into
stimulus events (target / lure / non-target), classifies responses (hit / miss /
omission / commission / etc.), adjusts response times across trial boundaries,
and aggregates per-block summary scores.
'''
import logging
import math
import os
import traceback
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from mend2np.utils import (
    setup_logger,
    parse_files,
    write_out,
    get_meta_cols,
    handle_multiple_responses,
    validate_params,
    run_task,
    copy_configured_columns,
)

# Module-level logger reference. `setup_logger` configures the same root logger,
# so helper functions in this module pick up the configured handlers regardless
# of which entry function (pgng / sert / etc.) is currently running.
logger = logging.getLogger('root')

# pgng has two run modes with different required JSON shapes:
#   formatted=False (raw PsychoPy/E-Prime output): needs metacols + blocks
#   formatted=True  (pre-cleaned tidy data):       needs cols
REQUIRED_PARAMS_UNFORMATTED = {
    'metacols': dict,
    'blocks': dict,
}
REQUIRED_PARAMS_FORMATTED = {
    'cols': dict,
}


def pgng(params:dict, formatted:bool=False, out:str=os.getcwd(), write:bool=True,
         filelist:str|list='', log=20, ind:bool=False, platform:str='psychopy') -> tuple:
    """Score one or more PGNG data files.

    :param params: configuration dict. See `tests/example_driver_pgng.py` for the unformatted shape
        (`metacols` + `blocks`) and `tests/example_driver_pgng_prefmt.py` for the formatted shape (`cols`).
    :param formatted: True if the input is already tidy with standard column names; default False.
    :param out: output directory.
    :param write: if True, write combined trials + scores CSVs.
    :param filelist: list of CSV paths, path to a text file with one CSV per line, or empty for GUI picker.
    :param log: log level.
    :param ind: if True, also write a per-file TSV for each input.
    :param platform: 'psychopy' (default) or 'eprime'.
    :returns: (combined_scores, combined_trials).
    """
    setup_logger(name='root', out=out, level=log).info('start')
    validate_params(params, REQUIRED_PARAMS_FORMATTED if formatted else REQUIRED_PARAMS_UNFORMATTED)

    def process_one(filepath, params, logger):
        filename_id = parse_files(filepath)
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)

        if not formatted:
            df = format_df(df, params, platform)

        if not check_cols(df):
            logger.error('columns misspecified, skipping this file')

        df = events_df(df)
        df = rt_adj(df)
        if check_timging_cols(df):
            df = onsets(df)

        df.insert(1, 'filename', filename)
        if ind:
            write_out(df, out, False, 'tsv')

        scores_row = pd.concat([get_meta_cols(df, params), score_df(df)], axis=1)
        scores_row.insert(1, 'filename', filename)
        return df, scores_row

    return run_task(
        params=params, filelist=filelist, out=out, write=write, log=log,
        process_file_fn=process_one,
    )

def check_cols(df:pd.DataFrame) -> bool:
    """Verify the formatted dataframe has every column scoring will require.

    Required columns: id, stimuli, response, rt, block, stim_targ_names, resp_key, type.
    Each must be present, have at least one non-null value, and pass type
    expectations (`rt` numeric, `stim_targ_names` cells are lists, `type` values
    are one of go/gng/gs). Missing or wrong-typed columns are logged as warnings.
    Returns True only when every check passed.
    """
    all_good = True

    # First pass: existence and non-emptiness for every required name.
    for var in ['id','stimuli','response','rt','block','stim_targ_names','resp_key','type']:
        if var not in df.columns:
            all_good = False
            logger.warning(f'required scoring column {var} not in dataset')
            continue
        elif not df[var].notnull().any():
            all_good = False
            logger.warning(f'required scoring column {var} is empty')
            continue

    if all_good:
        # additional checks
        if not is_numeric_dtype(df['rt']):
            all_good = False
            logger.warning(f'column "rt" is not numeric')

        # check if response & resp_key columns have compatible data types for comparison
        #if (is_numeric_dtype(df['response']) and is_string_dtype(df['rt'])) or \
        #(is_string_dtype(df['response']) and is_numeric_dtype(df['rt'])):
        #if not all(isinstance(x, type(df.loc[0,'response'])) for x in df.loc[0,'resp_key']):
        #    all_good = False
        #    logger.warning('columns "response" and "resp_key" have incompatible data types')

        if not isinstance(df['stim_targ_names'].head(1).values[0],list):
            all_good = False
            logger.warning('column "stim_targ_names" does not contain lists of target names')

        for s in df['type'].unique():
            if s not in ['go','gng','gs']:
                all_good = False
                logger.warning('values in the column "type" need to be one of ["go","gng","gs"]')
    
    return all_good
        

def check_timging_cols(df:pd.DataFrame) -> bool:
    """Verify the additional columns required for timing-relative onset calcs.

    Returns True only when `exp_start`, `stim_start`, `stim_dur` all exist,
    contain at least one non-null value, and are numeric. Missing columns are
    logged at DEBUG (not WARNING) because timing output is optional — the
    scorer happily skips it when the data don't support it.
    (Note: the function name is a historical typo of "timing"; preserved to
    avoid breaking external callers.)
    """

    timing_cols = ['exp_start','stim_start','stim_dur']
    all_good = True

    for var in timing_cols:
        if var not in df.columns:
            all_good = False
            logger.debug(f'column {var} not in dataset, will proceed without timing calcs')
            continue
        elif not df[var].notnull().any():
            all_good = False
            logger.debug(f'column {var} is empty, will proceed without timing calcs')
            continue
        if not is_numeric_dtype(df[var]):
            all_good = False
            logger.debug(f'column {var} is not numeric, will proceed without timing calcs')
        
    return all_good

def format_df(df:pd.DataFrame,params:dict,platform:str) -> pd.DataFrame:
    """Reshape a raw PsychoPy/E-Prime CSV into the library's standard column layout.

    The raw CSV typically has one set of columns per experimental block (e.g.
    `stimuli_1`, `block1_resp.keys`, `PGNGS_B1.thisTrialN`, ...). This function
    walks the `params['blocks']` config and, for each block, masks down to that
    block's trial rows and renames the per-block columns to the standard names
    the rest of the module uses (`stimuli`, `response`, `rt`, `trial`, etc.).
    The block index is appended as the `block` column, and per-participant
    metacols are broadcast onto every row.

    :param df: raw input dataframe (one CSV file's worth).
    :param params: config dict with `metacols` + `blocks`.
    :param platform: 'psychopy' or 'eprime'; affects mask construction and unit handling.
    :returns: the stacked, normalized dataframe with one row per trial across all blocks.
    """
    fmtdf = pd.DataFrame()

    for block in params['blocks']:

        tmpdf = pd.DataFrame()

        try:
            if platform == 'psychopy':
                mask = np.invert(df[params['blocks'][block]['cols']['trial']].isna())
            elif platform == 'eprime':
                mask = df[params['blocks'][block]['cols']['block']] == int(block)

            # Collect the per-participant metacol values now (so we know the schema),
            # but defer writing them into tmpdf until after the trial-level cols are
            # in place — assigning to an empty dataframe doesn't add rows, which used
            # to leave the last block's metacols permanently NaN in the trials output.
            metacol_values = {}
            for metacol in params['metacols']:
                if metacol.startswith('_'):
                    continue
                csv_col = params['metacols'][metacol]
                if not csv_col:
                    continue
                if csv_col in df.columns:
                    idx = df[csv_col].first_valid_index()
                    metacol_values[metacol] = df.at[idx, csv_col] if idx is not None else None
                else:
                    logger.warning(
                        f"metacols.{metacol}: configured CSV column '{csv_col}' is not in "
                        f"this file's columns — '{metacol}' will be missing for block {block}."
                    )

            # Per-block trial-level columns. `copy_configured_columns` warns when
            # a configured column name is missing from the CSV.
            copy_configured_columns(tmpdf, df, params['blocks'][block]['cols'],
                                    f'blocks.{block}.cols', mask=mask, logger=logger)

            # Now that tmpdf has rows (one per trial after the mask), broadcast each
            # per-participant metacol value across them, and reorder so metacols come
            # before the trial-level columns (matching the historical output layout).
            for metacol, value in metacol_values.items():
                tmpdf[metacol] = value
            ordered = list(metacol_values.keys()) + [c for c in tmpdf.columns if c not in metacol_values]
            tmpdf = tmpdf[ordered]
            
            for metavar in params['blocks'][block]['metavars']:
                if params['blocks'][block]['metavars'][metavar]:
                    if metavar == 'stim_targ_names':
                        tmpdf[metavar] = [params['blocks'][block]['metavars'][metavar]] * len(tmpdf)
                    # convert response key parameter to list
                    elif metavar == 'resp_key':
                        tmpdf[metavar] = [params['blocks'][block]['metavars'][metavar] if isinstance(params['blocks'][block]['metavars'][metavar], list) else [params['blocks'][block]['metavars'][metavar]]] * len(tmpdf)
                    else:
                        tmpdf[metavar] = params['blocks'][block]['metavars'][metavar]

            # Optional: stamp a single exp_start onto every row. Guard against a
            # totally-empty column — the original code did `.dropna().values[0]`
            # which IndexErrors when every row is NaN.
            if 'exp_start' in params['metacols']:
                exp_start_col = params['metacols']['exp_start']
                if exp_start_col and exp_start_col in df.columns:
                    valid = df[exp_start_col].dropna()
                    if len(valid) > 0:
                        tmpdf['exp_start'] = valid.values[0]

            # For Go/Stop blocks, the actual stop time supersedes the static stim_dur.
            if 'stop_time' in params['blocks'][block]['cols']:
                tmpdf['stim_dur'] = df.loc[mask,params['blocks'][block]['cols']['stop_time']]

            # handle multiple responses for touchscreen-based versions
            for resp_col in ['response','rt','rt_global']:
                if resp_col in tmpdf.columns:
                    # if string representation of a list, and list is not empty, keep only the first value of list
                    tmpdf[resp_col] = tmpdf[resp_col].apply(handle_multiple_responses)

            # clean & validate numeric columns
            for num_col in ['rt', 'exp_start', 'stim_start', 'stim_dur', 'rt_global']:
                if num_col in tmpdf.columns and is_string_dtype(tmpdf[num_col]):
                    tmpdf[num_col] = tmpdf[num_col].str.replace(r'[^\d.]', '', regex=True)
                    tmpdf[num_col] = pd.to_numeric(tmpdf[num_col], errors='coerce')

            # eprime: convert miliseconds to seconds, non-response rt as np.nan
            if platform == 'eprime':
                if 'exp_start' in tmpdf.columns:
                    tmpdf['exp_start'] = tmpdf['exp_start']/1000
                
                for rt_col in ['rt','rt_global']:
                    if rt_col in tmpdf.columns:
                        tmpdf[rt_col] = tmpdf[rt_col]/1000
                        for i, row in tmpdf.iterrows():
                            # E-Prime writes "no response" as exact zero (before the /1000 above); after the divide
                            # it is still float 0.0. Use a tolerance instead of == to be safe against any drift.
                            if tmpdf.loc[i,'response'] not in tmpdf.loc[i,'resp_key'] and abs(tmpdf.loc[i,rt_col]) < 1e-9:
                                tmpdf.loc[i,rt_col] = np.nan

            # If stim_start wasn't supplied, estimate it from exp_start + a static stim_dur
            # step. Requires `start_delta` (the pre-trial offset) to also be present —
            # if it isn't, log and skip estimation rather than KeyError.
            if (not 'stim_start' in tmpdf.columns
                    and 'exp_start' in tmpdf.columns
                    and 'stim_dur' in tmpdf.columns):
                if 'start_delta' in tmpdf.columns:
                    start = tmpdf['exp_start'].values[0] + tmpdf['start_delta'].values[0]
                    step = tmpdf['stim_dur'].values[0]
                    stim_start_vals = np.arange(start, start + len(tmpdf) * step, step)
                    tmpdf['stim_start'] = stim_start_vals[0:len(tmpdf)]
                else:
                    logger.debug(
                        "cannot estimate 'stim_start': 'start_delta' column not in tmpdf; "
                        "include a stim_start column in the config to suppress this."
                    )

            # For PsychoPy data, stamp the current block label onto every row. For
            # E-Prime data the block column was already brought across via the cols loop.
            if platform == 'psychopy':
                tmpdf['block'] = block

        except Exception as e:
            logger.error(f"{e}\n{traceback.format_exc()}\n")
            #print("see log file for errors")
            continue

        fmtdf = pd.concat([fmtdf,tmpdf],ignore_index=True)

    return fmtdf

def events_df(df:pd.DataFrame) -> pd.DataFrame:
    """Add `stim_class` and `resp_class` columns by event-classifying each trial.

    Operates block-by-block via `groupby('block').apply(event_block)` because the
    sequential state-machine classifiers (target/lure/hit/etc.) depend on the
    surrounding trials within the same block. Returns the dataframe with the
    new columns appended; the `block` column is reinserted right before `trial`
    in the output to keep the historical column layout.
    """

    df = df.set_index('block').groupby(level='block',as_index=False).apply(event_block).reset_index(drop=True)
    df.insert(df.columns.get_loc('trial'),'block',df.pop('block'))
    #dflt = onsets(dfl)
    #dflt.insert(dflt.columns.get_loc('trial'),'block',dflt.pop('block'))
    #dflt.drop(columns=dflt.filter(regex='level_.*',axis=1).columns.to_list(),axis=1,inplace=True)

    # for _, block in df.groupby('block'):
    #     if block['type'].values[0] == 'go':
    #         df['stim_class'] = block.apply(lambda x: 'target' if x['stimuli'] in x['stim_targ_names'] else '', axis=1)
    #         df['resp_class'] = resp_go(block)
    #     elif block['type'].values[0] == 'gng':
    #         df['stim_class'] = stim_gng(block)
    #         df['resp_class'] = resp_gng(block)
    #         #df_scores = pd.concat([df_scores,score_gng(block).reset_index(drop=True)],axis=1)
    #     elif block['type'].values[0] == 'gs':
    #         df['stim_class'] = stim_gs(block)
    #         df['resp_class'] = resp_gs(block)
    #     else:
    #         continue

    return df

def event_block(block:pd.DataFrame) -> pd.DataFrame:
    """Classify trials within a single block.

    Dispatches to the type-specific classifier pair based on the block's
    `type` value: go → `resp_go`; gng → `stim_gng` + `resp_gng`; gs →
    `stim_gs` + `resp_gs`. Returns the same block with new `stim_class` and
    `resp_class` columns.
    """
    # Work on an explicit copy with a fresh integer index. The original was
    # reset_index(inplace=True), which mutates the group pandas just handed us
    # inside groupby().apply() — that relied on internal pandas behaviour about
    # whether `apply` shares the group object with the caller.
    block = block.reset_index()
    block['stim_class'] = ''
    block['resp_class'] = ''

    if block['type'].values[0] == 'go':
        block['stim_class'] = block.apply(lambda x: 'target' if x['stimuli'] in x['stim_targ_names'] else '', axis=1)
        block['resp_class'] = resp_go(block)

    elif block['type'].values[0] == 'gng':
        block['stim_class'] = stim_gng(block)
        block['resp_class'] = resp_gng(block)
    
    elif block['type'].values[0] == 'gs':  # TODO: add stop var
        block['stim_class'] = stim_gs(block)
        block['resp_class'] = resp_gs(block)
        
        # = grp.apply(lambda x: 'lure' if x['stimuli'].shift(1) == "Stop.bmp" \
        #     else 'target' if x['stimuli'] in x['stim_targ_names'] else 'nontarget', axis=1)

    return block

def stim_gng(block:pd.DataFrame) -> pd.Series:
    """Classify each stimulus in a Go/No-go block as target, lure, or empty.

    The first occurrence of any target stimulus is a 'target' (the participant
    should respond); the second consecutive occurrence of the same target
    without an intervening different target is a 'lure' (they should withhold).
    A different target resets the "last seen" memory for all targets.
    """
    targs = block['stim_targ_names'].values[0]

    # last_seen tracks whether each target has been shown recently.
    # None = not seen recently → next occurrence is a 'target' (respond).
    # otherwise = seen on this row → next consecutive occurrence is a 'lure'.
    last_seen = dict(zip(targs, [None] * len(targs)))

    for i, current in block['stimuli'].items():
        if current in last_seen:
            block.at[i,'stim_class'] = 'target' if last_seen[current] is None else 'lure'
            last_seen = {key: None for key, value in last_seen.items()}
            last_seen[current] = i
        else:
            block.at[i,'stim_class'] = ''
    
    return block['stim_class']

def stim_gs(block:pd.DataFrame) -> pd.Series:
    """Classify each stimulus in a Go/Stop block as target or lure.

    A target followed on the very next trial by the configured stop stimulus
    becomes a 'lure' (the participant should withhold). Otherwise the target
    remains a 'target' (they should respond before the next stim window
    expires).
    """
    for i, row in block.iterrows():
        if block.loc[i,'stimuli'] in block.loc[i,'stim_targ_names']:
            next_is_stop = (i+1) in block.index and block.loc[i+1,'stimuli'] == block.loc[i+1,'stop']
            if next_is_stop:
                block.at[i,'stim_class'] = 'lure'
            else:
                block.at[i,'stim_class'] = 'target'

    return block['stim_class']


def _resp_match(block: pd.DataFrame, idx) -> bool:
    """True if `block.loc[idx, 'response']` is in `block.loc[idx, 'resp_key']`,
    or False if `idx` is out of bounds. Keeps the i+1/i+2 next-row lookups safe."""
    if idx not in block.index:
        return False
    return block.loc[idx, 'response'] in block.loc[idx, 'resp_key']


def resp_go(block:pd.DataFrame) -> pd.Series:
    """Classify responses within a Go block.

    On a target trial: 'hit' if the participant responded on that row or the
    immediately-following row, else 'om' (omission). On a non-target trial
    that follows a non-target trial: 'randcom' if the participant responded.
    Otherwise empty string.
    """
    for i, row in block.iterrows():
        if block.loc[i,'stim_class'] == 'target':
            if _resp_match(block, i) or _resp_match(block, i+1):
                block.loc[i,'resp_class'] = 'hit'
            else:
                block.loc[i,'resp_class'] = 'om'
        elif i>0 and block.loc[i-1,'stim_class'] != 'target' and _resp_match(block, i):
            block.loc[i,'resp_class'] = 'randcom'
        else:
            block.loc[i,'resp_class'] = ''

    return block['resp_class']

def resp_gng(block:pd.DataFrame) -> pd.Series:
    """Classify responses within a Go/No-go block.

    On a target trial: 'hit' / 'om' as in `resp_go`, but with a sticky `missed`
    flag set to True if the participant missed the most recent target.
    On a lure trial: 'mo' (miss-then-omit — they missed the prior target,
    making the lure response harder to interpret), 'com' (commission error —
    they responded to the lure), or 'rej' (correct rejection — no response).
    On a non-target non-lure trial after another non-target: 'randcom' if
    they responded, else empty.
    """
    # Tracks whether the most-recent target was missed. Resets to False on a hit.
    missed = False
    for i, row in block.iterrows():
        if block.loc[i,'stim_class'] == 'target':
            if _resp_match(block, i) or _resp_match(block, i+1):
                block.loc[i,'resp_class'] = 'hit'
                missed = False
            else:
                block.loc[i,'resp_class'] = 'om'
                missed = True
        elif block.loc[i,'stim_class'] == 'lure':
            if missed:
                block.loc[i,'resp_class'] = 'mo'
            elif _resp_match(block, i) or _resp_match(block, i+1):
                block.loc[i,'resp_class'] = 'com'
            else:
                block.loc[i,'resp_class'] = 'rej'
        elif i>0 and block.loc[i-1,'stim_class'] not in ['target','lure'] and _resp_match(block, i):
            block.loc[i,'resp_class'] = 'randcom'
        else:
            block.loc[i,'resp_class'] = ''

    return block['resp_class']

def resp_gs(block:pd.DataFrame) -> pd.Series:
    """Classify responses within a Go/Stop block.

    Like `resp_gng`, but the lure window for a Go/Stop trial extends across
    up to **two** trial rows after the lure stimulus — a commission counted
    if a response shows up in any of i, i+1, or i+2.
    Random commissions require two preceding non-target/non-lure rows
    (instead of one) for the same reason.
    """
    for i, row in block.iterrows():
        if block.loc[i,'stim_class'] == 'target':
            if _resp_match(block, i) or _resp_match(block, i+1):
                block.loc[i,'resp_class'] = 'hit'
            else:
                block.loc[i,'resp_class'] = 'om'
        elif block.loc[i,'stim_class'] == 'lure':
            if _resp_match(block, i) or _resp_match(block, i+1) or _resp_match(block, i+2):
                block.loc[i,'resp_class'] = 'com'
            else:
                block.loc[i,'resp_class'] = 'rej'
        elif i>1 and block.loc[i-1,'stim_class'] not in ['target','lure'] and block.loc[i-2,'stim_class'] not in ['target','lure'] \
            and _resp_match(block, i):
            block.loc[i,'resp_class'] = 'randcom'
        else:
            block.loc[i,'resp_class'] = ''

    return block['resp_class']

def rt_adj(df:pd.DataFrame) -> pd.DataFrame:
    """Compute an `rt_adj` column that adjusts response time for cross-row responses.

    PGNG response windows sometimes span multiple trial rows (especially for
    Go/Stop lures, whose response can land up to two trials later). The
    adjustment:
      - response on the current row: rt_adj = rt
      - response on the next row (target/lure): rt_adj = next_rt + offset to
        the next row (in absolute stim_start time if available, else in stim_dur)
      - gs-lure response on i+2: same idea, two trials ahead
      - otherwise rt_adj = NaN

    Each `i+1`/`i+2` lookup is bounds-checked so the last trial(s) of the
    overall dataframe don't trigger a KeyError.
    """
    df['rt_adj'] = np.nan

    for i, row in df.iterrows():

        # If no response was classified, there's nothing to compute.
        if df.loc[i,'resp_class'] == '':
            df.loc[i,'rt_adj'] = np.nan
            continue

        # 1) Response time recorded on this row → use as-is.
        if not np.isnan(df.loc[i,'rt']):
            df.loc[i,'rt_adj'] = df.loc[i,'rt']
            continue

        # 2) Target/lure with the response spilling onto the next row.
        next_in_bounds = (i+1) in df.index
        if (df.loc[i,'stim_class'] != ''
                and next_in_bounds
                and not np.isnan(df.loc[i+1,'rt'])):
            if 'stim_start' in df.columns:
                df.loc[i,'rt_adj'] = df.loc[i+1,'rt'] + (df.loc[i+1,'stim_start'] - df.loc[i,'stim_start'])
            else:
                df.loc[i,'rt_adj'] = df.loc[i+1,'rt'] + df.loc[i,'stim_dur']
            continue

        # 3) gs-lure with the response on i+2 (one row past the stop stim).
        plus2_in_bounds = (i+2) in df.index
        if (df.loc[i,'type'] == 'gs'
                and df.loc[i,'stim_class'] == 'lure'
                and plus2_in_bounds
                and not np.isnan(df.loc[i+2,'rt'])):
            if 'stim_start' in df.columns:
                df.loc[i,'rt_adj'] = df.loc[i+2,'rt'] + (df.loc[i+2,'stim_start'] - df.loc[i,'stim_start'])
            elif next_in_bounds:
                df.loc[i,'rt_adj'] = df.loc[i+2,'rt'] + df.loc[i,'stim_dur'] + df.loc[i+1,'stim_dur']
            continue

        df.loc[i,'rt_adj'] = np.nan

    return df

def onsets(df:pd.DataFrame) -> pd.DataFrame:
    """Compute trial-relative onset times for response events.

    Two new columns are added:
      - `stim_start_adj`: stimulus onset relative to experiment start
        (i.e. `stim_start - exp_start`).
      - `onsets`: the time at which the participant's response occurred
        relative to experiment start. For a hit recorded in the current row,
        this is `rt + stim_start - exp_start`. For a lure response that landed
        on the *next* trial's row (a common PGNG pattern when the response
        spills past the stim window), the next row's RT/stim_start are used
        instead. When neither a current-row nor next-row RT is available the
        stimulus onset itself is used as a fallback.
    """
    df['stim_start_adj'] = np.nan
    df['onsets'] = np.nan

    last_idx = df.index.max()
    for i, row in df.iterrows():
        if df.loc[i,'stim_start'] != '':
            df.loc[i,'stim_start_adj'] = df.loc[i,'stim_start'] - df.loc[i,'exp_start']
        if df.loc[i,'resp_class'] != '':
            if not np.isnan(df.loc[i,'rt']):
                df.loc[i,'onsets'] = df.loc[i,'rt'] + (df.loc[i,'stim_start'] - df.loc[i,'exp_start'])
            elif (df.loc[i,'stim_class'] != ''
                  and i < last_idx
                  and (i+1) in df.index
                  and not np.isnan(df.loc[i+1,'rt'])):
                # Response landed on the next trial's row; use the next row's timing.
                df.loc[i,'onsets'] = df.loc[i+1,'rt'] + (df.loc[i+1,'stim_start'] - df.loc[i+1,'exp_start'])
            else:
                df.loc[i,'onsets'] = df.loc[i,'stim_start'] - df.loc[i,'exp_start']
        else:
            df.loc[i,'onsets'] = np.nan

    return df

def score_df(df:pd.DataFrame) -> pd.DataFrame:
    """Build one row of per-file PGNG scores by horizontally concatenating per-block scores.

    Groups the per-trial dataframe by `block`, dispatches each block to the
    type-specific scorer (`score_go` / `score_gng` / `score_gs`), and joins
    every block's column set side-by-side into a single 1-row dataframe. Blocks
    with an unrecognized `type` value are skipped.
    """
    df_scores = pd.DataFrame(index=[0])

    for _, blk in df.groupby('block'):
        if blk['type'].values[0] == 'go':
            df_scores = pd.concat([df_scores,score_go(blk).reset_index(drop=True)],axis=1)
        elif blk['type'].values[0] == 'gng':
            df_scores = pd.concat([df_scores,score_gng(blk).reset_index(drop=True)],axis=1)
        elif blk['type'].values[0] == 'gs':
            df_scores = pd.concat([df_scores,score_gs(blk).reset_index(drop=True)],axis=1)
        else:
            # Unknown block type — log and skip instead of crashing.
            logger.warning(f"unknown block type {blk['type'].values[0]!r}; skipping")
            continue

    return df_scores

def _safe_ratio(numer: int, denom: int):
    """Return numer/denom, or NaN when denom == 0 instead of raising ZeroDivisionError."""
    return numer / denom if denom > 0 else np.nan


def score_go(block:pd.DataFrame):
    """Compute summary scores for a single Go block.

    Columns produced (prefix `go_{N}T_` where N is the target count):
      hit, om (omission), randcom (random commission), hit_rt_mean, hit_rt_sd,
      pctt (% of targets correctly hit).
    """
    stim_count = len(block['stim_targ_names'].head(1).values[0])
    n_hit = len(block.loc[block['resp_class']=='hit'])
    n_targets = len(block.loc[block['stim_class']=='target'])
    block_scores = pd.DataFrame({
            f'go_{stim_count}T_hit':n_hit,
            f'go_{stim_count}T_om':len(block.loc[block['resp_class']=='om']),
            f'go_{stim_count}T_randcom':len(block.loc[block['resp_class']=='randcom']),
            f'go_{stim_count}T_hit_rt_mean':block.loc[block['resp_class']=='hit','rt_adj'].mean(),
            f'go_{stim_count}T_hit_rt_sd':block.loc[block['resp_class']=='hit','rt_adj'].std(),
            f'go_{stim_count}T_pctt':_safe_ratio(n_hit, n_targets),
            },
            index=[0])

    return block_scores

def score_gng(block:pd.DataFrame):
    """Compute summary scores for a single Go/No-go block.

    Columns produced (prefix `gng_{N}T_`):
      hit, om, com (commission), rej (correct rejection), mo (miss-then-omit),
      randcom, hit_rt_mean, hit_rt_sd, com_rt_mean, com_rt_sd,
      pctt (% target hits), pcit (% inhibition trials correctly rejected).
    """
    stim_count = len(block['stim_targ_names'].head(1).values[0])
    n_hit = len(block.loc[block['resp_class']=='hit'])
    n_rej = len(block.loc[block['resp_class']=='rej'])
    n_targets = len(block.loc[block['stim_class']=='target'])
    n_lures = len(block.loc[block['stim_class']=='lure'])
    block_scores = pd.DataFrame({
            f'gng_{stim_count}T_hit':n_hit,
            f'gng_{stim_count}T_om':len(block.loc[block['resp_class']=='om']),
            f'gng_{stim_count}T_com':len(block.loc[block['resp_class']=='com']),
            f'gng_{stim_count}T_rej':n_rej,
            f'gng_{stim_count}T_mo':len(block.loc[block['resp_class']=='mo']),
            f'gng_{stim_count}T_randcom':len(block.loc[block['resp_class']=='randcom']),
            f'gng_{stim_count}T_hit_rt_mean':block.loc[block['resp_class']=='hit','rt_adj'].mean(),
            f'gng_{stim_count}T_hit_rt_sd':block.loc[block['resp_class']=='hit','rt_adj'].std(),
            f'gng_{stim_count}T_com_rt_mean':block.loc[block['resp_class']=='com','rt_adj'].mean(),
            f'gng_{stim_count}T_com_rt_sd':block.loc[block['resp_class']=='com','rt_adj'].std(),
            f'gng_{stim_count}T_pctt':_safe_ratio(n_hit, n_targets),
            f'gng_{stim_count}T_pcit':_safe_ratio(n_rej, n_lures),
            },
            index=[0])

    return block_scores

def score_gs(block:pd.DataFrame):
    """Compute summary scores for a single Go/Stop block.

    Columns produced (prefix `gs_{N}T_`):
      Same set as score_gng, plus stp_tm_rej / stp_tm_com (mean stop-signal
      time for correctly-rejected / commission-error lures).
    """
    stim_count = len(block['stim_targ_names'].head(1).values[0])
    n_hit = len(block.loc[block['resp_class']=='hit'])
    n_rej = len(block.loc[block['resp_class']=='rej'])
    n_targets = len(block.loc[block['stim_class']=='target'])
    n_lures = len(block.loc[block['stim_class']=='lure'])
    block_scores = pd.DataFrame({
            f'gs_{stim_count}T_hit':n_hit,
            f'gs_{stim_count}T_om':len(block.loc[block['resp_class']=='om']),
            f'gs_{stim_count}T_com':len(block.loc[block['resp_class']=='com']),
            f'gs_{stim_count}T_rej':n_rej,
            f'gs_{stim_count}T_randcom':len(block.loc[block['resp_class']=='randcom']),
            f'gs_{stim_count}T_hit_rt_mean':block.loc[block['resp_class']=='hit','rt_adj'].mean(),
            f'gs_{stim_count}T_hit_rt_sd':block.loc[block['resp_class']=='hit','rt_adj'].std(),
            f'gs_{stim_count}T_com_rt_mean':block.loc[block['resp_class']=='com','rt_adj'].mean(),
            f'gs_{stim_count}T_com_rt_sd':block.loc[block['resp_class']=='com','rt_adj'].std(),
            f'gs_{stim_count}T_stp_tm_rej':block.loc[block['resp_class']=='rej','stim_dur'].mean(),
            f'gs_{stim_count}T_stp_tm_com':block.loc[block['resp_class']=='com','stim_dur'].mean(),
            f'gs_{stim_count}T_pctt':_safe_ratio(n_hit, n_targets),
            f'gs_{stim_count}T_pcit':_safe_ratio(n_rej, n_lures),
            },
            index=[0])

    return block_scores

def cov_df(df:pd.DataFrame, window_duration:float):
    """Coefficient-of-variation over sliding time windows of `window_duration` seconds.

    Currently unused (the `cov` kwarg path in `pgng()` is commented out) but
    preserved for fMRI-style analyses that want per-window RT-mean/RT-SD/CoV.
    Within each block, walks contiguous windows from `stim_start_adj` of the
    first trial to the last trial's offset, and for each window that contains
    at least one hit, builds a row with: window bounds, hit count, mean RT,
    SD RT, and CoV (= SD / mean).
    """
    df_cov = pd.DataFrame()
    for _, block in df.groupby('block'):

        block_start = float(block['stim_start_adj'].head(1).values[0])
        block_end = float(block['stim_start_adj'].tail(1).values[0] + block['stim_dur'].head(1).values[0])
        block_duration = block_end - block_start
        num_windows = int(math.ceil(block_duration / window_duration))

        for i in range(0, num_windows):  # maybe add - 1
            window_start = block_start + (i * window_duration)
            window_end = block_start + ((i+1) * window_duration)

            window = block.loc[(block['resp_class'] == 'hit') & (block['onsets'] >= window_start) & (block['onsets'] <= window_end)]

            if window.empty:
                continue

            # build window row
            window_row = pd.DataFrame({
                # 'filename_id':window['filename_id'].head(1).values[0],
                # 'id':window['id'].head(1).values[0],
                # 'session':window['session'].head(1).values[0],
                # 'datetime':window['datetime'].head(1).values[0],
                # 'exp_name':window['exp_name'].head(1).values[0],
                'block':window['block'].head(1).values[0],
                'type':window['type'].head(1).values[0],
                'resp_class':window['resp_class'].head(1).values[0],
                'window_start':window_start,
                'window_end':window_end,
                'window_duration':window_end - window_start,
                'rt_count':len(window),
                'rt_mean':window['rt_adj'].mean(),
                'rt_sd':window['rt_adj'].std(),
                'rt_cov':window['rt_adj'].std() / window['rt_adj'].mean()
            },index=[0]).reset_index(drop=True)

            df_cov = pd.concat([df_cov,window_row],axis=0)

    return df_cov
