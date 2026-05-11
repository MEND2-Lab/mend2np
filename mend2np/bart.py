'''
BART (Balloon Analogue Risk Task) scoring.

Reads PsychoPy CSV output of the BART, formats per-trial fields (pump RT
deltas, popped flag, etc.) and computes participant-level summary scores
including total earnings, post-failure caution, and intertrial variability.
'''

import os
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

REQUIRED_PARAMS = {
    'metacols': dict,
    'cols': dict,
}


def bart(params:dict, out:str=os.getcwd(), write:bool=True, filelist:str|list='', formatted:bool=False, log=20,
         trial_filter:str='') -> tuple:
    """Score one or more BART data files.

    :param params: configuration dict mapping CSV columns to standard names; see `tests/example_driver_bart.py`.
    :param out: directory to write output CSVs (created if missing).
    :param write: if True, write combined trial- and score-level CSVs.
    :param filelist: list of CSV paths, path to a text file with one CSV path per line, or empty for GUI picker.
    :param formatted: True if the input is already in tidy form with standard column names; default False.
    :param log: log level (numeric or string).
    :param trial_filter: optional value of `trial_type` to restrict scoring to.
    :returns: (combined_scores, combined_trials).
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

    # handle multiple responses for touchscreen-based versions
    for resp_col in ['response', 'rt']:
        if resp_col in fmtdf.columns:
            fmtdf[resp_col] = fmtdf[resp_col].apply(lambda x: handle_multiple_responses(x, slice_index=slice(None)))

    # get delta between response times; guard against non-list / empty / NaN rows
    # (a participant who didn't pump on a trial has an empty list of click times)
    def _rt_deltas(lst):
        if not isinstance(lst, list) or len(lst) == 0:
            return []
        return [lst[0]] + [lst[i] - lst[i-1] for i in range(1, len(lst))]
    fmtdf['rt'] = fmtdf['rt'].apply(_rt_deltas)

    fmtdf['popped'] = fmtdf['popped'].astype(bool)
    return fmtdf


def score_df(df:pd.DataFrame, trial_filter):
    '''
    ntrials_popped: number of trials in which the balloon was popped
    ntrials_unpopped: number of trials in which the balloon was not popped
    ptrials_popped: proportion of trials in which the balloon was popped
    ptrials_unpopped: proportion of trials in which the balloon was not popped
    mean_pumps_popped: average number of pumps for popped trials
    mean_pumps_unpopped: average number of pumps for unpopped trials
    mean_rt_unpopped: average response time of all pumps in unpopped trials
    sd_rt_unpopped: standard deviation of response time of all pumps in unpopped trials
    total_earnings: total earnings
    mean_earnings: average earnings
    popped_ratio: ntrials_popped / ntrials_unpopped
    post_failure_mean_pumps: average pumps on trials after exploded balloons
    post_failure_mean_rt: average response time on trials after exploded balloons
    post_failure_sd_rt: standard deviation of response time on trials after exploded balloons
    intertrial_variability: standard deviation of total pumps divided by the mean of total pumps
    post_pumps_loss: average difference between the number of pumps on a loss trial and the immediate subsequent trial where participants elected to collect money prior to a balloon pop
    '''
    if trial_filter and 'trial_type' in df.columns:
        df = df.loc[df['trial_type'] == trial_filter]

    n_popped = int(df['popped'].sum())
    n_unpopped = int((~df['popped']).sum())
    n_total = len(df['popped'])
    popped_ratio = n_popped / n_unpopped if n_unpopped > 0 else np.nan
    popped_after_unpopped = df['popped'] & (df['popped'].shift(-1) == False)

    scores = pd.DataFrame({
        'ntrials_popped': n_popped,
        'ntrials_unpopped': n_unpopped,
        'popped_ratio': popped_ratio,
        'ptrials_popped': n_popped/n_total if n_total > 0 else np.nan,
        'ptrials_unpopped': n_unpopped/n_total if n_total > 0 else np.nan,
        'mean_pumps_popped': df.loc[df['popped'], 'nPumps'].mean(),
        'mean_pumps_unpopped': df.loc[~df['popped'], 'nPumps'].mean(),
        'mean_rt_unpopped': df.loc[~df['popped'], 'rt'].explode().mean(),
        'sd_rt_unpopped': df.loc[~df['popped'], 'rt'].explode().std(),
        'total_earnings': df['earnings'].sum(),
        'mean_earnings': df['earnings'].mean(),
        'intertrial_variability': df['nPumps'].std() / df['nPumps'].mean(),
        'post_failure_mean_pumps': df.shift(-1).loc[df['popped'], 'nPumps'].mean(),
        'post_failure_mean_rt': df.shift(-1).loc[df['popped'], 'rt'].explode().mean(),
        'post_failure_sd_rt': df.shift(-1).loc[df['popped'], 'rt'].explode().std(),
        'post_pumps_loss': (df['nPumps'] - df['nPumps'].shift(-1)).loc[popped_after_unpopped].mean(),
    }, index=[0])

    return scores
