'''

'''

import os
import sys
import traceback
import pandas as pd
import numpy as np
from mend2np.utils import setup_logger, select_files, write_out, get_meta_cols, handle_multiple_responses

def bart(params:dict, out:str=os.getcwd(), filelist:str|list='', formatted:bool=False,log=20,
         verbose:bool=False):

    os.makedirs(out, exist_ok=True)

    global logger
    logger = setup_logger(name='root',out=out,level=log,verbose=verbose)
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

            df.insert(1,'filename',filename)

            combined_trials = pd.concat([combined_trials,df],axis=0,ignore_index=True)

            this_row = pd.concat([get_meta_cols(df,params),score_df(df)],axis=1)
            this_row.insert(1,'filename',filename)
            combined_scores = pd.concat([combined_scores,this_row],axis=0,ignore_index=True)

        except Exception as e:
            logger.error(f'{filename} : {e}\n{traceback.format_exc()}\n')

    if not combined_trials.empty:
        write_out(combined_trials,out,True,'csv','trials')
    if not combined_scores.empty:
        write_out(combined_scores,out,True,'csv','scores')
    
    logger.info('end')

    return combined_scores

def format_df(df:pd.DataFrame,params:dict) -> pd.DataFrame:
    '''
    '''

    fmtdf = pd.DataFrame()

    mask = np.invert(df[params['cols']['trial']].isna())

    for metacol in params['metacols']:
        if params['metacols'][metacol]:
            fmtdf[metacol] = df.loc[mask,params['metacols'][metacol]]

    for col in params['cols']:
        if params['cols'][col]:
            fmtdf[col] = df.loc[mask,params['cols'][col]]

    #TODO: add quality checks, data type, empty, missing cols

    # handle multiple responses for touchscreen-based versions
    for resp_col in ['response','rt']:
        if resp_col in fmtdf.columns:
            # if string representation of a list, and list is not empty, keep only the first value of list
            fmtdf[resp_col] = fmtdf[resp_col].apply(handle_multiple_responses)

    fmtdf['popped'] = fmtdf['popped'].astype(bool)

    return fmtdf

def score_df(df:pd.DataFrame):
    '''
    ntrials_popped: number of trials in which the balloon was popped
    ntrials_unpopped: number of trials in which the balloon was not popped
    ptrials_popped: proportion of trials in which the balloon was popped
    ptrials_unpopped: proportion of trials in which the balloon was not popped
    mean_pumps_popped: average number of pumps for popped trials
    mean_pumps_unpopped: average number of pumps for unpopped trials
    total_earnings: total earnings
    mean_earnings: average earnings
    popped_ratio: ntrials_popped / ntrials_unpopped
    post_failure_mean_pumps: average pumps on trials after exploded balloons
    intertrial_variability: standard deviation of total pumps divided by the mean of total pumps
    post_pumps_loss: calculated by averaging the difference between the number of pumps on a loss trial and the immediate subsequent trial where participants elected to collect money prior to a balloon pop
    '''

    scores = pd.DataFrame({
        'ntrials_popped':sum(df['popped']),
        'ntrials_unpopped':sum(~df['popped']),
        'popped_ratio':sum(df['popped'])/sum(~df['popped']),
        'ptrials_popped':sum(df['popped'])/len(df['popped']),
        'ptrials_unpopped':sum(~df['popped'])/len(df['popped']),
        'mean_pumps_popped':df.loc[df['popped'],'nPumps'].mean(),
        'mean_pumps_unpopped':df.loc[~df['popped'],'nPumps'].mean(),
        'total_earnings':df['earnings'].sum(),
        'mean_earnings':df['earnings'].mean(),
        'intertrial_variability':df.loc[:,'nPumps'].std() / df.loc[:,'nPumps'].mean(),
        'post_failure_mean_pumps':df.shift(-1).loc[df['popped'],'nPumps'].mean(),
        'post_pumps_loss': (df['nPumps'].shift(-1) - df['nPumps']).loc[df['popped'] & (df['popped'].shift(-1) == False)].mean()
    },
    index=[0])

    return scores
