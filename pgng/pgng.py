'''

'''
import pandas as pd
import os
import numpy as np
import traceback
import sys
import math
import logging
from collections.abc import Iterable
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
import re
from ast import literal_eval
from datetime import datetime
from tkinter import filedialog as fd

def pgng(params:dict,formatted:bool=False,score=True,cov_window:float=np.nan,out:str=os.getcwd(),filelist:str|list='',log=20):
    '''
    '''

    # make output directory
    os.makedirs(out, exist_ok=True)

    # initiate logger
    global logger
    logger = setup_logger(name='root',out=out,level=log)
    logger.info("start")
    
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
    if not np.isnan(cov_window):
        combined_cov = pd.DataFrame()

    # loop through data files
    for filepath in filepaths:
        #print(filepath)
        logger.info(f'processing: {filepath}')

        # putting everything in a try block & write errors to error_log
        try:
            filename_id = parse_files(filepath)
            filename = os.path.basename(filepath)
            
            # read data
            df = pd.read_csv(filepath)

            # if the data aren't already formatted properly
            if not formatted:
                df = format_df(df,params)

            if not check_cols(df):
                logger.error('columns misspecified, skipping this file')

            # add event columns
            df = events_df(df)
            
            # add adjusted rt column
            df = rt_adj(df)

            # add onset columns
            if check_timging_cols(df):
                df = onsets(df)

            df.insert(1,'filename',filename)
            write_out(df,out,False,'tsv')

            combined_trials = pd.concat([combined_trials,df],axis=0,ignore_index=True)
        
            #TODO write formatted onsets file

            # make some score output
            if score:
                this_row = pd.concat([get_meta_cols(df,params),score_df(df)],axis=1)
                combined_scores = pd.concat([combined_scores,this_row],axis=0,ignore_index=True)
            if not np.isnan(cov_window):
                this_row = pd.concat([get_meta_cols(df,params),cov_df(df,window_duration=cov_window)],axis=1,ignore_index=True)
                combined_cov = pd.concat([combined_cov,this_row],axis=0,ignore_index=True)

        except Exception as e:
            logger.error(f'{filename} : {e}\n{traceback.format_exc()}\n')
            print("see log file for errors")
            continue
        
        try:
            if not combined_trials.empty:
                write_out(combined_trials,out,True,'csv','trials')
            if score and not combined_scores.empty:
                write_out(combined_scores,out,True,'csv','scores')
            if not np.isnan(cov_window) and not combined_cov.empty:
                write_out(combined_cov,out,True,'csv',f'cov_{cov_window}')
        except Exception as e:
            logger.error(f'{filename} : {e}\n{traceback.format_exc()}\n')
            print("see log file for errors")

    
    logger.info('end')
    
    if score:
        return combined_scores

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

def parse_files(filepath:str) -> tuple:
    # parse file name into useful bits
    basename = os.path.basename(filepath)
    base = basename.rsplit('.', 1)[0]  
    parts = base.split('_')
    date_str = parts[-2] + '_' + parts[-1]
    for fmt in ["%Y-%m-%d_%Hh%M.%S.%f","%m-%d-%Y_%Hh%M.%S.%f"]:
        try:
            dt = datetime.strptime(date_str,fmt)
            break
        except ValueError:
            pass
    id = re.match(r'^[^_]+',basename).group(0)

    #return (id,dt,basename)
    return (id)

def write_out(df:pd.DataFrame,out:str,merged:bool,filetype:str,tag:str=''):

    if filetype == 'csv':
        sep = ','
    elif filetype == 'tsv':
        sep = '\t'

    if merged:
        if 'exp_name' in df.columns:
            exp_name = str(df['exp_name'].head(1).values[0]).replace(os.sep,'')
        else:
            exp_name = 'PGNG'
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

def check_cols(df:pd.DataFrame) -> bool:
    '''
    required: id, stimuli, response, rt, block, stim_targ_names, resp_key, type
    '''

    all_good = True

    # check exist & not empty
    for var in ['id','stimuli','response','rt','block','stim_targ_names','resp_key','type']:
        if var not in df.columns:
            all_good = False
            logger.warning(f'required scoring column {var} not in dataset')
            continue
        elif not df[var].notnull().any():
            all_good = False
            logger.warning(f'required scoring column {var} is empty')
            continue

    # additional checks
    if not is_numeric_dtype(df['rt']):
        all_good = False
        logger.warning(f'column "rt" is not numeric')

    # check if reponse & rt columns have compatible data types for comparison
    if (is_numeric_dtype(df['response']) and is_string_dtype(df['rt'])) or \
    (is_string_dtype(df['response']) and is_numeric_dtype(df['rt'])):
        all_good = False
        logger.warning(f'columns "response" and "resp_key" have incompatible data types\n \
                       response dtype: {df["response"].dtype}, rt dtype: {df["resp_key"].dtype}')

    if not isinstance(df['stim_targ_names'].head(1).values[0],list):
        all_good = False
        logger.warning(f'column "stim_targ_names" does not contain lists of target names')

    for s in df['type'].unique():
        if s not in ['go','gng','gs']:
            all_good = False
            logger.warning('values in the column "type" need to be one of ["go","gng","gs"]')
    
    return all_good
        

def check_timging_cols(df:pd.DataFrame) -> bool:
    '''
    addtional required for timing: exp_start, stim_start, stim_dur
    '''

    # TODO: make it possible to use stim_start OR stim_dur

    all_good = True

    for var in ['exp_start','stim_start','stim_dur']:
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

def format_df(df:pd.DataFrame,params:dict) -> pd.DataFrame:
    '''
    '''

    fmtdf = pd.DataFrame()

    for block in params['blocks']:

        tmpdf = pd.DataFrame()

        try:
            mask = np.invert(df[params['blocks'][block]['cols']['trial']].isna())

            for metacol in params['metacols']:
                if params['metacols'][metacol]:
                    tmpdf[metacol] = df.loc[mask,params['metacols'][metacol]]
            
            for col in params['blocks'][block]['cols']:
                if params['blocks'][block]['cols'][col]:
                    tmpdf[col] = df.loc[mask,params['blocks'][block]['cols'][col]]
            
            for metavar in params['blocks'][block]['metavars']:
                if params['blocks'][block]['metavars'][metavar]:
                    if metavar == 'stim_targ_names':
                        tmpdf[metavar] = [params['blocks'][block]['metavars'][metavar]] * len(tmpdf)
                    else:
                        tmpdf[metavar] = params['blocks'][block]['metavars'][metavar]

            if 'exp_start' in params['metacols']:
                tmpdf['exp_start'] = df[params['metacols']['exp_start']].dropna().values[0]

            # for gs, update stim_dur to correct times
            if 'stop_time' in params['blocks'][block]['cols']:
                tmpdf['stim_dur'] = df.loc[mask,params['blocks'][block]['cols']['stop_time']]

            # handle multiple responses for touchscreen-based versions
            for resp_col in ['response','rt']:
                if resp_col in tmpdf.columns:
                    # if string representation of a list, and list is not empty, keep only the first value of list
                    tmpdf[resp_col] = tmpdf[resp_col].apply(handle_multiple_responses)

            # clean & validate numeric columns
            for num_col in ['rt', 'exp_start', 'stim_start', 'stim_dur']:
                if num_col in tmpdf.columns and is_string_dtype(tmpdf[num_col]):
                    tmpdf[num_col] = tmpdf[num_col].str.replace(r'[^\d.]', '', regex=True)
                    tmpdf[num_col] = pd.to_numeric(tmpdf[num_col], errors='coerce')

            tmpdf['block'] = block

        except Exception as e:
            logger.error(f"{e}\n{traceback.format_exc()}\n")
            print("see log file for errors")
            continue

        fmtdf = pd.concat([fmtdf,tmpdf],ignore_index=True)

    return fmtdf
    
def handle_multiple_responses(value) -> str|float|int|None:
    # if string representation of a list, and list is not empty, keep only the first value of list
    if isinstance(value,str) and re.match(r'^\[.*\]$',value):
        eval_value = literal_eval(value)
        if isinstance(eval_value,list):
            if len(eval_value) > 0:
                return eval_value[0]
            else:
                return None
    else:
        return value

def get_meta_cols(df,params):
    '''
    for aggregated values (one row per participant) collect meta variables into one row
    '''

    metacols_df = pd.DataFrame(index=[0])

    for metacol in params['metacols']:
        if params['metacols'][metacol]:
            metacols_df[metacol] = df[metacol].head(1).values[0]

    return metacols_df.reset_index(drop=True)

def events_df(df:pd.DataFrame) -> pd.DataFrame:
    '''
    label rows as PGNGS event types
    takes in a formatted dataset
    '''

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
    '''
    '''

    block.reset_index(inplace=True)
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
    '''
    '''

    targs = block['stim_targ_names'].values[0]

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
    '''
    '''
    
    #targs = block['stim_targ_names'].values[0]

    for i, row in block.iterrows():
        if block.loc[i,'stimuli'] in block.loc[i,'stim_targ_names']:
            if block.loc[i+1,'stimuli'] == block.loc[i+1,'stop']:
                block.at[i,'stim_class'] = 'lure'
            else:
                block.at[i,'stim_class'] = 'target'
    
    return block['stim_class']


def resp_go(block:pd.DataFrame) -> pd.Series:
    '''
    '''

    for i, row in block.iterrows():
        if block.loc[i,'stim_class'] == 'target':
            if block.loc[i,'response'] == block.loc[i,'resp_key'] or block.loc[i+1,'response'] == block.loc[i+1,'resp_key']:
                block.loc[i,'resp_class'] = 'hit'
            else:
                block.loc[i,'resp_class'] = 'om'
        elif i>0 and block.loc[i-1,'stim_class'] != 'target' and block.loc[i,'response'] == block.loc[i,'resp_key']:
            block.loc[i,'resp_class'] = 'randcom'
        else:
            block.loc[i,'resp_class'] = ''

    return block['resp_class']

def resp_gng(block:pd.DataFrame) -> pd.Series:
    '''
    '''
    missed = False
    for i, row in block.iterrows():
        if block.loc[i,'stim_class'] == 'target':
            if block.loc[i,'response'] == block.loc[i,'resp_key'] or block.loc[i+1,'response'] == block.loc[i+1,'resp_key']:
                block.loc[i,'resp_class'] = 'hit'
                missed = False
            else:
                block.loc[i,'resp_class'] = 'om'
                missed = True
        elif block.loc[i,'stim_class'] == 'lure':
            if missed:
                block.loc[i,'resp_class'] = 'mo'
            elif block.loc[i,'response'] == block.loc[i,'resp_key'] or block.loc[i+1,'response'] == block.loc[i+1,'resp_key']:
                block.loc[i,'resp_class'] = 'com'
            else:
                block.loc[i,'resp_class'] = 'rej'
        elif i>0 and block.loc[i-1,'stim_class'] not in ['target','lure'] and block.loc[i,'response'] == block.loc[i,'resp_key']:
            block.loc[i,'resp_class'] = 'randcom'
        else:
            block.loc[i,'resp_class'] = ''
            
    return block['resp_class']

def resp_gs(block:pd.DataFrame) -> pd.Series:
    '''
    '''
    for i, row in block.iterrows():
        if block.loc[i,'stim_class'] == 'target':
            if block.loc[i,'response'] == block.loc[i,'resp_key'] or block.loc[i+1,'response'] == block.loc[i+1,'resp_key']:
                block.loc[i,'resp_class'] = 'hit'
            else:
                block.loc[i,'resp_class'] = 'om'
        elif block.loc[i,'stim_class'] == 'lure':
            if block.loc[i,'response'] == block.loc[i,'resp_key'] or block.loc[i+1,'response'] == block.loc[i+1,'resp_key'] \
                or block.loc[i+2,'response'] == block.loc[i+2,'resp_key']:
                block.loc[i,'resp_class'] = 'com'
            else:
                block.loc[i,'resp_class'] = 'rej'
        elif i>1 and block.loc[i-1,'stim_class'] not in ['target','lure'] and block.loc[i-2,'stim_class'] not in ['target','lure'] \
            and block.loc[i,'response'] == block.loc[i,'resp_key']:
            block.loc[i,'resp_class'] = 'randcom'
        else:
            block.loc[i,'resp_class'] = ''
            
    return block['resp_class']

def rt_adj(df:pd.DataFrame) -> pd.DataFrame:
    '''
    '''
    df['rt_adj'] = np.nan

    for i, row in df.iterrows():

        # if there's some response
        if df.loc[i,'resp_class'] != '':

            # if there's a response time in this row
            if not np.isnan(df.loc[i,'rt']):
                df.loc[i,'rt_adj'] = df.loc[i,'rt']

            # if this is a target or lure and a response time in the next row
            elif df.loc[i,'stim_class'] != '' and not np.isnan(df.loc[i+1,'rt']):
                if 'stim_start' in df.columns:
                    df.loc[i,'rt_adj'] = df.loc[i+1,'rt'] + (df.loc[i+1,'stim_start'] - df.loc[i,'stim_start'])
                else:
                    df.loc[i,'rt_adj'] = df.loc[i+1,'rt'] + df.loc[i,'stim_dur']
            
            # if this is a gs lure with response in the row after 'stop'
            elif df.loc[i,'type'] == 'gs' and df.loc[i,'stim_class'] == 'lure' and not np.isnan(df.loc[i+2,'rt']):
                if 'stim_start' in df.columns:
                    df.loc[i,'rt_adj'] = df.loc[i+2,'rt'] + (df.loc[i+2,'stim_start'] - df.loc[i,'stim_start'])
                else:
                    df.loc[i,'rt_adj'] = df.loc[i+2,'rt'] + df.loc[i,'stim_dur'] + df.loc[i+1,'stim_dur']
            else:
                df.loc[i,'rt_adj'] = np.nan
        else:
            df.loc[i,'rt_adj'] = np.nan

    return df

def onsets(df:pd.DataFrame) -> pd.DataFrame:
    #TODO
    '''
    '''
    df['stim_start_adj'] = np.nan
    df['onsets'] = np.nan
    #df['rt_adj'] = np.nan

    for i, row in df.iterrows():
        if df.loc[i,'stim_start'] != '':
            df.loc[i,'stim_start_adj'] = df.loc[i,'stim_start'] - df.loc[i,'exp_start']
        if df.loc[i,'resp_class'] != '':
            if not np.isnan(df.loc[i,'rt']):
                df.loc[i,'onsets'] = df.loc[i,'rt'] + (df.loc[i,'stim_start'] - df.loc[i,'exp_start'])
                #df.loc[i,'rt_adj'] = df.loc[i,'rt']
            elif df.loc[i,'stim_class'] != '' and not np.isnan(df.loc[i+1,'rt']):
                df.loc[i,'onsets'] = df.loc[i+1,'rt'] + (df.loc[i+1,'stim_start'] - df.loc[i+1,'exp_start'])
                #df.loc[i,'rt_adj'] = df.loc[i+1,'rt'] + (df.loc[i+1,'stim_start'] - df.loc[i,'stim_start'])
            else:
                df.loc[i,'onsets'] = df.loc[i,'stim_start'] - df.loc[i,'exp_start']
        else:
            df.loc[i,'onsets'] = np.nan

    return df

def score_df(df:pd.DataFrame) -> pd.DataFrame:
    '''
    '''
    # # setup meta columns
    # df_scores = pd.DataFrame({
    #         'filename':df['filename'].head(1).values[0],
    #         'id':df['id'].head(1).values[0],
    #         'session':df['session'].head(1).values[0],
    #         'datetime':df['datetime'].head(1).values[0],
    #         'exp_name':df['exp_name'].head(1).values[0],
    #         'software_version':df['software_version'].head(1).values[0],
    #         'framerate':df['framerate'].head(1).values[0]
    #         },
    #         index=[0]).reset_index(drop=True)

    df_scores = pd.DataFrame(index=[0])
    
    for _, blk in df.groupby('block'):
        if blk['type'].values[0] == 'go':
            df_scores = pd.concat([df_scores,score_go(blk).reset_index(drop=True)],axis=1)
        elif blk['type'].values[0] == 'gng':
            df_scores = pd.concat([df_scores,score_gng(blk).reset_index(drop=True)],axis=1)
        elif blk['type'].values[0] == 'gs':
            df_scores = pd.concat([df_scores,score_gs(blk).reset_index(drop=True)],axis=1)
        else:
            #TODO add error logging
            continue
    
    return df_scores

def score_go(block:pd.DataFrame):
    '''
    '''    
    stim_count = len(block['stim_targ_names'].head(1).values[0])
    block_scores = pd.DataFrame({
            f'go_{stim_count}T_hit':len(block.loc[block['resp_class']=='hit']),
            f'go_{stim_count}T_om':len(block.loc[block['resp_class']=='om']),
            f'go_{stim_count}T_randcom':len(block.loc[block['resp_class']=='randcom']),
            f'go_{stim_count}T_hit_rt_mean':block.loc[block['resp_class']=='hit','rt_adj'].mean(),
            f'go_{stim_count}T_hit_rt_sd':block.loc[block['resp_class']=='hit','rt_adj'].std(),
            f'go_{stim_count}T_pctt':len(block.loc[block['resp_class']=='hit']) / len(block.loc[block['stim_class']=='target'])
            },
            index=[0])
    
    return block_scores

def score_gng(block:pd.DataFrame):
    '''
    '''
    stim_count = len(block['stim_targ_names'].head(1).values[0])
    block_scores = pd.DataFrame({
            f'gng_{stim_count}T_hit':len(block.loc[block['resp_class']=='hit']),
            f'gng_{stim_count}T_om':len(block.loc[block['resp_class']=='om']),
            f'gng_{stim_count}T_com':len(block.loc[block['resp_class']=='com']),
            f'gng_{stim_count}T_rej':len(block.loc[block['resp_class']=='rej']),
            f'gng_{stim_count}T_mo':len(block.loc[block['resp_class']=='mo']),
            f'gng_{stim_count}T_randcom':len(block.loc[block['resp_class']=='randcom']),
            f'gng_{stim_count}T_hit_rt_mean':block.loc[block['resp_class']=='hit','rt_adj'].mean(),
            f'gng_{stim_count}T_hit_rt_sd':block.loc[block['resp_class']=='hit','rt_adj'].std(),
            f'gng_{stim_count}T_com_rt_mean':block.loc[block['resp_class']=='com','rt_adj'].mean(),
            f'gng_{stim_count}T_com_rt_sd':block.loc[block['resp_class']=='com','rt_adj'].std(),
            f'gng_{stim_count}T_pctt':len(block.loc[block['resp_class']=='hit']) / len(block.loc[block['stim_class']=='target']),
            f'gng_{stim_count}T_pcit':len(block.loc[block['resp_class']=='rej']) / len(block.loc[block['stim_class']=='lure'])
            },
            index=[0])
    
    return block_scores

def score_gs(block:pd.DataFrame):
    '''
    '''
    stim_count = len(block['stim_targ_names'].head(1).values[0])
    block_scores = pd.DataFrame({
            f'gs_{stim_count}T_hit':len(block.loc[block['resp_class']=='hit']),
            f'gs_{stim_count}T_om':len(block.loc[block['resp_class']=='om']),
            f'gs_{stim_count}T_com':len(block.loc[block['resp_class']=='com']),
            f'gs_{stim_count}T_rej':len(block.loc[block['resp_class']=='rej']),
            f'gs_{stim_count}T_randcom':len(block.loc[block['resp_class']=='randcom']),
            f'gs_{stim_count}T_hit_rt_mean':block.loc[block['resp_class']=='hit','rt_adj'].mean(),
            f'gs_{stim_count}T_hit_rt_sd':block.loc[block['resp_class']=='hit','rt_adj'].std(),
            f'gs_{stim_count}T_com_rt_mean':block.loc[block['resp_class']=='com','rt_adj'].mean(),
            f'gs_{stim_count}T_com_rt_sd':block.loc[block['resp_class']=='com','rt_adj'].std(),
            f'gs_{stim_count}T_stp_tm_rej':block.loc[block['resp_class']=='rej','stim_dur'].mean(),
            f'gs_{stim_count}T_stp_tm_com':block.loc[block['resp_class']=='com','stim_dur'].mean(),
            f'gs_{stim_count}T_pctt':len(block.loc[block['resp_class']=='hit']) / len(block.loc[block['stim_class']=='target']),
            f'gs_{stim_count}T_pcit':len(block.loc[block['resp_class']=='rej']) / len(block.loc[block['stim_class']=='lure'])
            },
            index=[0])
    
    return block_scores

def cov_df(df:pd.DataFrame,window_duration:float):
    '''
    coefficient of variation
    '''
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
