'''
FEPT (Facial Emotion Perception Task) scoring.

Reads PsychoPy CSV output of the FEPT, classifies each stimulus filename into
emotion / race / sex / animal categories via configurable regex maps, and
emits per-category accuracy, RT, and misclassification counts.
'''

import pandas as pd
import numpy as np
import os
import re
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
    'blocks': dict,
}


def fept(params:dict, formatted:bool=False, out:str=os.getcwd(), write:bool=True, filelist:str|list='',
         log:int|str=20, ind:bool=False):
    """Score one or more FEPT data files.

    :param params: configuration dict (see `tests/example_driver_fept.py`).
    :param formatted: True if the input is already tidy with standard column names.
    :param out: output directory.
    :param write: if True, write the combined scores CSV.
    :param filelist: list of CSV paths, path to a text file with one CSV per line, or empty for GUI picker.
    :param log: log level.
    :param ind: if True, also write a per-file CSV for each input.
    :returns: combined_scores dataframe.
    """
    setup_logger(name='root', out=out, level=log).info('start')
    validate_params(params, REQUIRED_PARAMS)

    def process_one(filepath, params, logger):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        if not formatted:
            df = format_df(df, params)
            df.insert(1, 'filename', filename)
            if ind:
                write_out(df, out, False, 'csv')
        scores_row = pd.concat([get_meta_cols(df, params), score_df(df)], axis=1)
        scores_row.insert(1, 'filename', filename)
        return None, scores_row  # fept historically writes only scores, not trials

    combined_scores, _ = run_task(
        params=params, filelist=filelist, out=out, write=write, log=log,
        process_file_fn=process_one,
        write_trials=False, write_scores=True,
    )
    return combined_scores


def format_df(df:pd.DataFrame,params:dict) -> pd.DataFrame:
    fmtdf = pd.DataFrame()

    for block in params['blocks']:

        tmpdf = pd.DataFrame()

        mask = np.invert(df[params['blocks'][block]['cols']['stimuli']].isna())

        for metacol in params['metacols']:
            if params['metacols'][metacol]:
                tmpdf[metacol] = df.loc[mask,params['metacols'][metacol]]

        for metavar in params['blocks'][block]['metavars']:
            if params['blocks'][block]['metavars'][metavar]:
                tmpdf[metavar] = params['blocks'][block]['metavars'][metavar]

        for col in params['blocks'][block]['cols']:
            if params['blocks'][block]['cols'][col]:
                tmpdf[col] = df.loc[mask,params['blocks'][block]['cols'][col]]

        if 'key_labels' in params['blocks'][block].keys():
            tmpdf['response_label'] = tmpdf['response'].map(params['blocks'][block]['key_labels'])
            tmpdf['correct_response_label'] = tmpdf['correct_response'].map(params['blocks'][block]['key_labels'])

        if 'stim_class_map' in params['blocks'][block].keys():
            stim_class_map = params['blocks'][block]['stim_class_map']
            # compile regexes once per block
            regex_map = {
                category: re.compile("(" + "|".join(map(re.escape, keys)) + ")")
                for category, keys in stim_class_map.items()
            }

            tmpdf['stim_class'] = tmpdf['stimuli'].apply(lambda x: parse_stimulus_filename(x,regex_map,stim_class_map))

        # handle multiple responses for touchscreen-based versions
        for resp_col in ['response','rt']:
            if resp_col in tmpdf.columns:
                # if string representation of a list, and list is not empty, keep only the first value of list
                tmpdf[resp_col] = tmpdf[resp_col].apply(handle_multiple_responses)

        tmpdf['block'] = block

        fmtdf = pd.concat([fmtdf,tmpdf],ignore_index=True)

    return fmtdf


def parse_stimulus_filename(filename,regex_map:dict,stim_class_map:dict,sep:str=';'):
    base = os.path.splitext(os.path.basename(filename))[0]
    classes = []
    for category, pattern in regex_map.items():
        match = pattern.search(base)
        if match:
            abbr = match.group(1)
            mapped_val = stim_class_map[category].get(abbr)
        else:
            mapped_val = None
        if mapped_val:
            classes.append(mapped_val)
    return sep.join(classes)


def score_df(df:pd.DataFrame):
    df_scores = pd.DataFrame(index=[0])

    for _, blk in df.groupby('block'):

        # global scores
        df_scores = pd.concat([df_scores,score_blk(blk).reset_index(drop=True)],axis=1)

        # expand stim_class into separate category_N columns on a fresh copy
        category_columns = blk['stim_class'].str.split(';',expand=True)
        category_columns.columns = [f'category_{i}' for i in range(category_columns.shape[1])]
        blk_with_cats = pd.concat([blk.reset_index(drop=True), category_columns.reset_index(drop=True)], axis=1)

        # scores per category
        for category_column in category_columns.columns:
            for _, blk_cat in blk_with_cats.groupby(category_column):
                cat_name = blk_cat[category_column].head(1).values[0]
                df_scores = pd.concat([df_scores,score_blk(blk_cat,cat_name).reset_index(drop=True)],axis=1)

        # classification errors
        for _, emot_type in blk_with_cats.groupby('correct_response'):
            error_trials = emot_type.loc[emot_type['response']!=emot_type['correct_response']]
            for _, wrong_emot in error_trials.groupby('response',dropna=False):
                misclassification_label = f'{wrong_emot["type"].head(1).values[0]}_{wrong_emot["correct_response_label"].head(1).values[0]}_as_{wrong_emot["response_label"].head(1).values[0]}'
                df_scores[misclassification_label] = len(wrong_emot)

        for _, error_type in blk_with_cats.loc[blk_with_cats['response']!=blk_with_cats['correct_response']].groupby('response'):
            false_alarm_label = f'{error_type["type"].head(1).values[0]}_{error_type["response_label"].head(1).values[0]}_false_alarm'
            df_scores[false_alarm_label] = len(error_type)

    return df_scores


def score_blk(block,cat_name=None):
    if cat_name:
        blk_type = f'{block["type"].head(1).values[0]}_{cat_name}'
    else:
        blk_type = block['type'].head(1).values[0]

    block = block.copy()
    block['rt_global'] = block['stimulus_duration'] + block['mask_duration'] + block['rt']
    n_correct = len(block.loc[block['response']==block['correct_response']])
    n_incorrect = len(block.loc[block['response']!=block['correct_response']])
    denom = n_correct + n_incorrect
    blk_scores = pd.DataFrame({
        f'{blk_type}_correct_ct': n_correct,
        f'{blk_type}_incorrect_ct': n_incorrect,
        f'{blk_type}_acc': n_correct / denom if denom > 0 else np.nan,
        f'{blk_type}_rt_mean': block['rt'].mean(),
        f'{blk_type}_rt_sd': block['rt'].std(),
        f'{blk_type}_correct_rt_mean': block.loc[block['response']==block['correct_response'],'rt'].mean(),
        f'{blk_type}_correct_rt_sd': block.loc[block['response']==block['correct_response'],'rt'].std(),
        f'{blk_type}_incorrect_rt_mean': block.loc[block['response']!=block['correct_response'],'rt'].mean(),
        f'{blk_type}_incorrect_rt_sd': block.loc[block['response']!=block['correct_response'],'rt'].std(),
    },index=[0])

    return blk_scores
