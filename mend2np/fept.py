'''

'''

import pandas as pd
import numpy as np
import os
import re
import traceback
import sys
from mend2np.utils import setup_logger, select_files, write_out, get_meta_cols, handle_multiple_responses

def fept(params:dict, formatted:bool=False, out:str=os.getcwd(), filelist:str|list='', log:int|str=20, ind:bool=False):
    '''
    '''

    # make output directory
    os.makedirs(out, exist_ok=True)

    # initiate logger
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
    #combined_trials = pd.DataFrame()
    combined_scores = pd.DataFrame()

    # loop through data files
    for filepath in filepaths:
        #print(filepath)
        logger.info(f'processing: {filepath}')

        try:
            filename = os.path.basename(filepath)

            df = pd.read_csv(filepath)

            if not formatted:
                df = format_df(df,params)
                df.insert(1,'filename',filename)
                if ind:
                    write_out(df,out,False,'csv')

            this_row = pd.concat([get_meta_cols(df,params),score_df(df)],axis=1)
            this_row.insert(1,'filename',filename)
            combined_scores = pd.concat([combined_scores,this_row],axis=0,ignore_index=True)

        except Exception as e:
            logger.error(f'{filename} : {e}\n{traceback.format_exc()}\n')
            #print("see log file for errors")
            continue
    
    try:
        # if not combined_trials.empty:
        #     write_out(combined_trials,out,True,'csv','trials')
        if not combined_scores.empty:
            write_out(combined_scores,out,True,'csv','scores')
    except Exception as e:
        logger.error(f'{filename} : {e}\n{traceback.format_exc()}\n')
        #print("see log file for errors")

    return combined_scores

def format_df(df:pd.DataFrame,params:dict) -> pd.DataFrame:
    '''
    '''

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
            # compile regexes
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
        
        # TODO add validation

        tmpdf['block'] = block

        fmtdf = pd.concat([fmtdf,tmpdf],ignore_index=True)

    return fmtdf

def parse_stimulus_filename(filename,regex_map:dict,stim_class_map:dict,sep:str=';'):
    '''
    '''
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
    '''
    '''
    df_scores = pd.DataFrame(index=[0])

    for _, blk in df.groupby('block'):

        # global scores
        df_scores = pd.concat([df_scores,score_blk(blk).reset_index(drop=True)],axis=1)

        # expand stim_class
        category_columns = blk['stim_class'].str.split(';',expand=True)
        category_columns.columns = [f'category_{i}' for i in range(category_columns.shape[1])]
        blk = pd.concat([blk, category_columns], axis=1)

        # scores per category
        for category_column in category_columns.columns:
            for _, blk_cat in blk.groupby(category_column):
                cat_name = blk_cat[category_column].head(1).values[0]
                df_scores = pd.concat([df_scores,score_blk(blk_cat,cat_name).reset_index(drop=True)],axis=1)

        # classification errors
        for _, emot_type in blk.groupby('correct_response'):
            error_trials = emot_type.loc[emot_type['response']!=emot_type['correct_response']]
            for _, wrong_emot in error_trials.groupby('response',dropna=False):
                misclassification_label = f'{wrong_emot["type"].head(1).values[0]}_{wrong_emot["correct_response_label"].head(1).values[0]}_as_{wrong_emot["response_label"].head(1).values[0]}'
                df_scores[misclassification_label] = len(wrong_emot)

        for _, error_type in blk.loc[blk['response']!=blk['correct_response']].groupby('response'):
            false_alarm_label = f'{error_type["type"].head(1).values[0]}_{error_type["response_label"].head(1).values[0]}_false_alarm'
            df_scores[false_alarm_label] = len(error_type)

    return df_scores

def score_blk(block,cat_name=None):
    '''
    '''
    if cat_name:
        blk_type = f'{block["type"].head(1).values[0]}_{cat_name}'
    else:
        blk_type = block['type'].head(1).values[0]

    block['rt_global'] = block['stimulus_duration'] + block['mask_duration'] + block['rt']
    blk_scores = pd.DataFrame({
        # overall correct count
        f'{blk_type}_correct_ct':len(block.loc[block['response']==block['correct_response']]),
        # overall incorrect count
        f'{blk_type}_incorrect_ct':len(block.loc[block['response']!=block['correct_response']]),
        # overall accuracy
        f'{blk_type}_acc':len(block.loc[block['response']==block['correct_response']]) / \
                        (len(block.loc[block['response']==block['correct_response']]) + \
                        len(block.loc[block['response']!=block['correct_response']])),
        # overall rt mean
        f'{blk_type}_rt_mean':block['rt'].mean(),
        # overall rt sd
        f'{blk_type}_rt_sd':block['rt'].std(),
        # correct response rt mean
        f'{blk_type}_correct_rt_mean':block.loc[block['response']==block['correct_response'],'rt'].mean(),
        # correct response rt sd
        f'{blk_type}_correct_rt_sd':block.loc[block['response']==block['correct_response'],'rt'].std(),
        # incorrect response rt mean
        f'{blk_type}_incorrect_rt_mean':block.loc[block['response']!=block['correct_response'],'rt'].mean(),
        # incorrect response rt sd
        f'{blk_type}_incorrect_rt_sd':block.loc[block['response']!=block['correct_response'],'rt'].std(),
    },index=[0])

    return blk_scores
