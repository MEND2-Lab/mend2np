'''

'''
import re
import os
import sys
import traceback
import pandas as pd
import numpy as np
from math import ceil
from pathlib import Path
from mend2np.utils import setup_logger, select_files, write_out, get_meta_cols, handle_multiple_responses

touch_resp_mapping = {
    'LeftImage':1,
    'MiddleImage':2,
    'RightImage':3
}

def sert(params:dict, out:str=os.getcwd(), write:bool=True, filelist:str|list='', formatted:bool=False, log=20,
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

            df = parse_choice_columns(df)

            df = add_switch_rep(df)

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

    fmtdf['response'] = fmtdf['response'].apply(lambda x: [touch_resp_mapping.get(resp, resp) for resp in x] if isinstance(x, list) else touch_resp_mapping.get(x, x))

    return fmtdf

def add_switch_rep(df:pd.DataFrame) -> pd.DataFrame:
    df['block'] = ((df['trial']) // 10) + 1
    df['block_nunique_cues'] = df.groupby('block')['cue'].transform('nunique')
    df['block_switch_rep'] = np.where(df['block_nunique_cues'] > 1, 'switch', 'repeat')
    return df

def parse_choice_value(value:str) -> dict:
    if pd.isna(value):
        return {'class': None, 'type': None, 'color': None, 'shape': None}

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
        'shape': 'rect' if shape == 'rectangle' else shape
    }

def parse_choice_columns(df:pd.DataFrame) -> pd.DataFrame:
    for side in ['left', 'middle', 'right']:
        choice_col = f'{side}_choice'
        if choice_col not in df.columns:
            continue

        parsed = df[choice_col].apply(parse_choice_value).apply(pd.Series)
        parsed.columns = [f'{choice_col}_{suffix}' for suffix in parsed.columns]
        df = pd.concat([df, parsed], axis=1)

    return df

def parse_responses(df:pd.DataFrame) -> pd.DataFrame:
    for i, row in df.iterrows():
        
        this_response = row['response'] if 'response' in row else None
        this_rt = row['rt'] if 'rt' in row else None
        this_correct_resp = int(row['correct_resp']) if 'correct_resp' in row else None

        if isinstance(this_response, list):
            # for each response in list, if it's a digit convert to int, else map using touch_resp_mapping, if not in mapping keep as is
            this_response = [int(resp) if str(resp).isdigit() else touch_resp_mapping.get(resp, resp) for resp in this_response]
        else:
            if not this_response is None:
                if str(this_response).isdigit():
                    this_response = int(this_response)
                elif str(this_response).isalpha():
                    this_response = [touch_resp_mapping.get(this_response, this_response)]
                else:
                    this_response = [this_response]
            else:
                this_response = []

        if isinstance(this_rt, list):
            this_rt = [float(rt) for rt in this_rt]
        else:
            this_rt = [float(this_rt)] if not this_rt is None else []

        df.at[i,'num_responses'] = len(this_response)

        if df.at[i,'num_responses'] > 0:

            df.at[i,'first_response'] = this_response[0]
            df.at[i,'first_response_rt'] = this_rt[0]

            df.at[i,'last_response'] = this_response[-1]
            df.at[i,'last_response_rt'] = this_rt[-1]

            if this_correct_resp in this_response:
                df.at[i,'correct'] = 1
                correct_index = this_response.index(this_correct_resp)
                df.at[i,'correct_resp_index'] = correct_index
                df.at[i,'correct_resp_rt'] = this_rt[correct_index]
            else:
                df.at[i,'correct'] = 0
                df.at[i,'correct_resp_rt'] = None

            # if df.at[i,'num_responses'] > 1:
            #     df.at[i,'multiple_responses'] = 1
            #     if this_response[-1] == this_correct_resp:
            #         df.at[i,'last_response_correct'] = 1
            #     else:                
            #         df.at[i,'last_response_correct'] = 0
            # else:
            #     df.at[i,'multiple_responses'] = 0
            #     df.at[i,'last_response_correct'] = df.at[i,'correct']
        else:
            df.at[i,'first_response'] = None
            df.at[i,'first_response_rt'] = None
            df.at[i,'last_response'] = None
            df.at[i,'last_response_rt'] = None
            df.at[i,'correct'] = 0
            df.at[i,'correct_resp_rt'] = None
            # df.at[i,'multiple_responses'] = 0
            # df.at[i,'last_response_correct'] = 0

    return df

def safe_diff(sdict: dict, out_key: str, a_key: str, b_key: str):
    if a_key in sdict and b_key in sdict:
        sdict[out_key] = sdict[a_key] - sdict[b_key]
    else:
        sdict[out_key] = np.nan

def score_df(df:pd.DataFrame,trial_filter:str) -> pd.DataFrame:
    '''
    '''
    score_dict = {}

    # apply trial filter if specified
    if trial_filter:
        df = df.query(trial_filter)

    for event_type, event_group in df.groupby('event_type'):

        score_dict[f'{event_type}_num_trials'] = len(event_group)
        score_dict[f'{event_type}_num_correct'] = event_group['correct'].sum()
        score_dict[f'{event_type}_accuracy'] = event_group['correct'].mean()
        score_dict[f'{event_type}_mean_first_rt'] = event_group['first_response_rt'].mean()
        score_dict[f'{event_type}_median_first_rt'] = event_group['first_response_rt'].median()
        score_dict[f'{event_type}_std_first_rt'] = event_group['first_response_rt'].std()
        score_dict[f'{event_type}_mean_correct_resp_rt'] = event_group.loc[event_group['correct']==1,'correct_resp_rt'].mean()
        score_dict[f'{event_type}_median_correct_resp_rt'] = event_group.loc[event_group['correct']==1,'correct_resp_rt'].median()
        score_dict[f'{event_type}_std_correct_resp_rt'] = event_group.loc[event_group['correct']==1,'correct_resp_rt'].std()

        for switch_rep, switch_rep_group in event_group.groupby('block_switch_rep'):
            score_dict[f'{event_type}_{switch_rep}_num_trials'] = len(switch_rep_group)
            score_dict[f'{event_type}_{switch_rep}_num_correct'] = switch_rep_group['correct'].sum()
            score_dict[f'{event_type}_{switch_rep}_accuracy'] = switch_rep_group['correct'].mean()
            score_dict[f'{event_type}_{switch_rep}_mean_first_rt'] = switch_rep_group['first_response_rt'].mean()
            score_dict[f'{event_type}_{switch_rep}_median_first_rt'] = switch_rep_group['first_response_rt'].median()
            score_dict[f'{event_type}_{switch_rep}_std_first_rt'] = switch_rep_group['first_response_rt'].std()
            score_dict[f'{event_type}_{switch_rep}_mean_correct_resp_rt'] = switch_rep_group.loc[switch_rep_group['correct']==1,'correct_resp_rt'].mean()
            score_dict[f'{event_type}_{switch_rep}_median_correct_resp_rt'] = switch_rep_group.loc[switch_rep_group['correct']==1,'correct_resp_rt'].median()
            score_dict[f'{event_type}_{switch_rep}_std_correct_resp_rt'] = switch_rep_group.loc[switch_rep_group['correct']==1,'correct_resp_rt'].std()

            for cue, cue_group in switch_rep_group.groupby('cue'):
                score_dict[f'{event_type}_{switch_rep}_{cue}_num_trials'] = len(cue_group)
                score_dict[f'{event_type}_{switch_rep}_{cue}_num_correct'] = cue_group['correct'].sum()
                score_dict[f'{event_type}_{switch_rep}_{cue}_accuracy'] = cue_group['correct'].mean()
                score_dict[f'{event_type}_{switch_rep}_{cue}_mean_first_rt'] = cue_group['first_response_rt'].mean()
                score_dict[f'{event_type}_{switch_rep}_{cue}_median_first_rt'] = cue_group['first_response_rt'].median()
                score_dict[f'{event_type}_{switch_rep}_{cue}_std_first_rt'] = cue_group['first_response_rt'].std()
                score_dict[f'{event_type}_{switch_rep}_{cue}_mean_correct_resp_rt'] = cue_group.loc[cue_group['correct']==1,'correct_resp_rt'].mean()
                score_dict[f'{event_type}_{switch_rep}_{cue}_median_correct_resp_rt'] = cue_group.loc[cue_group['correct']==1,'correct_resp_rt'].median()
                score_dict[f'{event_type}_{switch_rep}_{cue}_std_correct_resp_rt'] = cue_group.loc[cue_group['correct']==1,'correct_resp_rt'].std()
        
        safe_diff(score_dict, f'{event_type}_switch_cost_mean_first_rt', f'{event_type}_switch_mean_first_rt', f'{event_type}_repeat_mean_first_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_median_first_rt', f'{event_type}_switch_median_first_rt', f'{event_type}_repeat_median_first_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_mean_correct_resp_rt', f'{event_type}_switch_mean_correct_resp_rt', f'{event_type}_repeat_mean_correct_resp_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_median_correct_resp_rt', f'{event_type}_switch_median_correct_resp_rt', f'{event_type}_repeat_median_correct_resp_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_accuracy', f'{event_type}_switch_accuracy', f'{event_type}_repeat_accuracy')
        safe_diff(score_dict, f'{event_type}_switch_cost_num_correct', f'{event_type}_switch_num_correct', f'{event_type}_repeat_num_correct')

        safe_diff(score_dict, f'{event_type}_switch_cost_color_mean_first_rt', f'{event_type}_switch_color_mean_first_rt', f'{event_type}_repeat_color_mean_first_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_color_median_first_rt', f'{event_type}_switch_color_median_first_rt', f'{event_type}_repeat_color_median_first_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_color_mean_correct_resp_rt', f'{event_type}_switch_color_mean_correct_resp_rt', f'{event_type}_repeat_color_mean_correct_resp_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_color_median_correct_resp_rt', f'{event_type}_switch_color_median_correct_resp_rt', f'{event_type}_repeat_color_median_correct_resp_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_color_accuracy', f'{event_type}_switch_color_accuracy', f'{event_type}_repeat_color_accuracy')
        safe_diff(score_dict, f'{event_type}_switch_cost_color_num_correct', f'{event_type}_switch_color_num_correct', f'{event_type}_repeat_color_num_correct')

        safe_diff(score_dict, f'{event_type}_switch_cost_shape_mean_first_rt', f'{event_type}_switch_shape_mean_first_rt', f'{event_type}_repeat_shape_mean_first_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_shape_median_first_rt', f'{event_type}_switch_shape_median_first_rt', f'{event_type}_repeat_shape_median_first_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_shape_mean_correct_resp_rt', f'{event_type}_switch_shape_mean_correct_resp_rt', f'{event_type}_repeat_shape_mean_correct_resp_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_shape_median_correct_resp_rt', f'{event_type}_switch_shape_median_correct_resp_rt', f'{event_type}_repeat_shape_median_correct_resp_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_shape_accuracy', f'{event_type}_switch_shape_accuracy', f'{event_type}_repeat_shape_accuracy')
        safe_diff(score_dict, f'{event_type}_switch_cost_shape_num_correct', f'{event_type}_switch_shape_num_correct', f'{event_type}_repeat_shape_num_correct')

        safe_diff(score_dict, f'{event_type}_switch_cost_lethality_mean_first_rt', f'{event_type}_switch_lethality_mean_first_rt', f'{event_type}_repeat_lethality_mean_first_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_lethality_median_first_rt', f'{event_type}_switch_lethality_median_first_rt', f'{event_type}_repeat_lethality_median_first_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_lethality_mean_correct_resp_rt', f'{event_type}_switch_lethality_mean_correct_resp_rt', f'{event_type}_repeat_lethality_mean_correct_resp_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_lethality_median_correct_resp_rt', f'{event_type}_switch_lethality_median_correct_resp_rt', f'{event_type}_repeat_lethality_median_correct_resp_rt')
        safe_diff(score_dict, f'{event_type}_switch_cost_lethality_accuracy', f'{event_type}_switch_lethality_accuracy', f'{event_type}_repeat_lethality_accuracy')
        safe_diff(score_dict, f'{event_type}_switch_cost_lethality_num_correct', f'{event_type}_switch_lethality_num_correct', f'{event_type}_repeat_lethality_num_correct')

    return pd.DataFrame(score_dict, index=[0])