'''
'''

import os
import re
import logging
import pandas as pd
from datetime import datetime
from tkinter import filedialog as fd
from ast import literal_eval

def setup_logger(name:str='root', out:str='out', level:int|str=20, verbose:bool=False):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    datetime_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(module)s : %(message)s')
    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(out,f'log_{datetime_string}.log'),mode='w')
    file_handler.setFormatter(formatter)
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

def write_out(df:pd.DataFrame,out:str,merged:bool,filetype:str,tag:str='',exp_name:str=''):

    if filetype == 'csv':
        sep = ','
    elif filetype == 'tsv':
        sep = '\t'

    if merged:
        
        if len(exp_name) == 0 and 'exp_name' in df.columns:
            exp_name = str(df['exp_name'].head(1).values[0]).replace(os.sep,'')
            
        filename = f"{exp_name}_n{df['id'].nunique()}_{tag}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{filetype}"
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

def handle_multiple_responses(value, slice_index=0) -> str|float|int|None:
    # if string representation of a list, and list is not empty, keep only the first value of list
    if isinstance(value,str) and re.match(r'^\[.*\]$',value):
        eval_value = literal_eval(value)
        if isinstance(eval_value,list):
            if len(eval_value) > 0:
                return eval_value[slice_index]
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