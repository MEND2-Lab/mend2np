'''

'''

import os
import sys
import logging
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from tkinter import filedialog as fd

def bart(params:dict,out:str=os.getcwd(),filelist:str|list='',formatted:bool=False,score:bool=True,log=20):

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
    if score:
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

            if score:
                combined_scores = pd.concat([combined_scores,score_df(df)],axis=0,ignore_index=True)

        except Exception as e:
            logger.error(f'{filename} : {e}\n{traceback.format_exc()}\n')

    if not combined_trials.empty:
        write_out(combined_trials,out,True,'csv','trials')
    if score and not combined_scores.empty:
        write_out(combined_scores,out,True,'csv','scores')
    
    logger.info('end')


def setup_logger(name,out,level):
    datetime_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(module)s : %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(out,f'log_{datetime_string}.log'),mode='w')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

def select_files() -> tuple:
    filepaths = fd.askopenfilenames(
        title='Select CSV files to score',
        filetypes=(("CSV Files", "*.csv"),),
        initialdir=os.getcwd(),
        multiple=True)
    return filepaths

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

    fmtdf['popped'] = fmtdf['popped'].astype(bool)

    return fmtdf

def score_df(df:pd.DataFrame):
    '''
    number popped
    number not popped
    proportion popped
    proportion not popped
    average number of pumps for popped trials
    average number of pumps for unpopped trials
    total earnings
    average earnings
    '''

    #TODO: add flexible handling of meta columns

    scores = pd.DataFrame({
        'filename':df['filename'].head(1).values[0],
        'id':df['id'].head(1).values[0],
        'session':df['session'].head(1).values[0],
        'datetime':df['datetime'].head(1).values[0],
        'exp_name':df['exp_name'].head(1).values[0],
        'software_version':df['software_version'].head(1).values[0],
        'framerate':df['framerate'].head(1).values[0],
        'ntrials_popped':sum(df['popped']),
        'ntrials_unpopped':sum(~df['popped']),
        'ptrials_popped':sum(df['popped'])/len(df['popped']),
        'ptrials_unpopped':sum(~df['popped'])/len(df['popped']),
        'mean_pumps_popped':df.loc[df['popped'],'nPumps'].mean(),
        'mean_pumps_unpopped':df.loc[~df['popped'],'nPumps'].mean(),
        'total_earnings':df['earnings'].sum(),
        'mean_earnings':df['earnings'].mean()
    },
    index=[0])

    return scores

def write_out(df:pd.DataFrame,out:str,merged:bool,filetype:str,tag:str=''):

    if filetype == 'csv':
        sep = ','
    elif filetype == 'tsv':
        sep = '\t'

    if merged:
        if 'exp_name' in df.columns:
            exp_name = str(df['exp_name'].head(1).values[0]).replace(os.sep,'')
        else:
            exp_name = 'BART'
        filename = f"{exp_name}_n{df['id'].nunique()}_{tag}.{filetype}"
        df.to_csv(os.path.join(out,filename),index=False,sep=sep)

    else:
        filename = ''
        for var in ['filename_id','id','session','exp_name','datetime']:
            if var in df.columns:
                if not filename:
                    filename =  ''.join([filename,str(df[var].head(1).values[0]).replace(os.sep,'')])
                else:
                    filename = '_'.join([filename,str(df[var].head(1).values[0]).replace(os.sep,'')])
        filename = filename + f'.{filetype}'
        df.to_csv(os.path.join(out,filename),index=False,sep=sep)
