
import re
import os
import sys
import traceback
import pandas as pd
import numpy as np
from math import ceil
from pathlib import Path
from mend2np.utils import setup_logger, select_files, write_out, get_meta_cols, handle_multiple_responses

resp_mapping = {
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

def synonyms(params:dict, out:str=os.getcwd(), write:bool=True, filelist:str|list='', formatted:bool=False, log=20,
         trial_filter:str='') -> tuple:

    os.makedirs(out, exist_ok=True)

    global logger
    logger = setup_logger(name='root',out=out,level=log)
    logger.info('start')

    # sort how the file list was passed
    if filelist:
        if isinstance(filelist, list):
            # if filelist is iterable
            filepaths = filelist
        elif os.path.isfile(filelist):
            # else if a file, try reading filepaths
            try:
                filepaths = [line.strip() for line in open(filelist, 'r', encoding='utf-8')]
            except Exception as e:
                logger.critical(f'problem reading filelist: {filelist}: {e}\n{traceback.format_exc()}\n')
                sys.exit(1)
        else:
            logger.critical(f'problem with filelist: {filelist}, consult docs or leave blank to use GUI file select')
            sys.exit(1)
    else:
        # else do the GUI file select
        filepaths = select_files()

    # initiate combined files
    combined_trials = pd.DataFrame()
    combined_scores = pd.DataFrame()

    # loop through data files
    for filepath in filepaths:
        logger.info(f'processing: {filepath}')

        try:
            filename = os.path.basename(filepath)

            df = pd.read_csv(filepath)

            if not formatted:
                df = format_df(df,params)

            df = parse_responses(df)

            df.insert(1,'filename',filename)

            combined_trials = pd.concat([combined_trials,df],axis=0,ignore_index=True)

            this_row = pd.concat([get_meta_cols(df,params),score_df(df,trial_filter)],axis=1)
            this_row.insert(1,'filename',filename)
            combined_scores = pd.concat([combined_scores,this_row],axis=0,ignore_index=True)

        except Exception as e:
            logger.error(f'{filename} : {e}\n{traceback.format_exc()}\n')

    if write:
        if not combined_trials.empty:
            write_out(combined_trials,out,True,'csv','trials')
        if not combined_scores.empty:
            write_out(combined_scores,out,True,'csv','scores')
    
    logger.info('end')

    return combined_scores, combined_trials

def format_df(df:pd.DataFrame,params:dict) -> pd.DataFrame:

    fmtdf = pd.DataFrame()

    mask = np.invert(df[params['cols']['trial']].isna())

    for metacol in params['metacols']:
        if params['metacols'][metacol] and params['metacols'][metacol] in df.columns:
            fmtdf[metacol] = df.loc[mask,params['metacols'][metacol]]

    for col in params['cols']:
        if params['cols'][col] and params['cols'][col] in df.columns:
            fmtdf[col] = df.loc[mask,params['cols'][col]]

     # handle multiple responses
    for resp_col in ['response','rt']:
        if resp_col in fmtdf.columns:
            # if string representation of a list, convert to list
            fmtdf[resp_col] = fmtdf[resp_col].apply(lambda x: handle_multiple_responses(x, slice_index=slice(None)))

    for opt_col in ['response','correct_resp']:
        if opt_col in fmtdf.columns:
            fmtdf[opt_col] = fmtdf[opt_col].apply(lambda x: [resp_mapping.get(resp, resp) for resp in x] if isinstance(x, list) else resp_mapping.get(x, x))

    return fmtdf

def parse_responses(df:pd.DataFrame):

    for i, row in df.iterrows():
        
        this_response = row['response'] if 'response' in row else None
        this_rt = row['rt'] if 'rt' in row else None
        this_correct_resp = int(row['correct_resp']) if 'correct_resp' in row else None

        if isinstance(this_response, list):
            #this_response = [resp for resp in this_response if resp is not None]
            this_response = [int(resp) if str(resp).isdigit() else resp_mapping.get(resp, resp) for resp in this_response]
        else:
            if str(this_response).isdigit():
                this_response = [int(this_response)]
            elif not this_response is None:
                this_response = [resp_mapping.get(this_response, this_response)]
            else:
                this_response = []

        if isinstance(this_rt, list):
            this_rt = [float(rt) for rt in this_rt]
        else:
            this_rt = [float(this_rt)] if not this_rt is None else []

        df.at[i,'num_responses'] = len(this_response)

        if df.at[i,'num_responses'] > 0:

            df.at[i,'response_last'] = this_response[-1]
            df.at[i,'rt_last'] = this_rt[-1]

            if this_correct_resp in this_response:
                df.at[i,'correct'] = 1
                correct_index = this_response.index(this_correct_resp)
                df.at[i,'correct_resp_index'] = correct_index
            else:
                df.at[i,'correct'] = 0
        else:
            df.at[i,'correct'] = 0


    return df

def score_df(df:pd.DataFrame,trial_filter:str) -> pd.DataFrame:
    '''
    '''
    score_dict = {}

    # apply trial filter if specified
    if trial_filter:
        df = df.query(trial_filter)

    score_dict[f'num_correct'] = df['correct'].sum()
    score_dict[f'prop_correct'] = df['correct'].mean()
    score_dict[f'mean_rt'] = df['rt_last'].mean()
    score_dict[f'sd_rt'] = df['rt_last'].std()
    score_dict[f'mean_correct_resp_rt'] = df.loc[df['correct']==1,'rt_last'].mean()
    score_dict[f'std_correct_resp_rt'] = df.loc[df['correct']==1,'rt_last'].std()
    score_dict[f'mean_incorrect_resp_rt'] = df.loc[df['correct']==0,'rt_last'].mean()
    score_dict[f'std_incorrect_resp_rt'] = df.loc[df['correct']==0,'rt_last'].std()

    return pd.DataFrame(score_dict, index=[0])
