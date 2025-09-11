'''

'''
import pandas as pd
from . import utils
import os
import numpy as np
import traceback
import sys
import math

def main(params:dict,formatted:bool=False,score:bool=True,cov:bool=False,out:str=os.getcwd(),filelist:str=''):
    '''
    '''

    os.makedirs(out, exist_ok=True)

    global error_log

    # temporary error log, add proper logging later
    error_log = os.path.join(out,'error_log.csv')
    if os.path.exists(error_log):
        os.remove(error_log)

    if filelist:
        try:
            filepaths = [line.strip() for line in open(filelist, 'r', encoding='utf-8')]
        except Exception as e:
            print(f'problem reading file list: {filelist}: {e}')
            sys.exit(1)
    else:
        filepaths = utils.select_files()

    if score:
        combined_scores = pd.DataFrame()
    if cov:
        window = 60
        combined_cov = pd.DataFrame()

    for filepath in filepaths:
        print(filepath)
        filename_id = utils.parse_files(filepath)
        filename = os.path.basename(filepath)

        try:
            df = pd.read_csv(filepath)
            fmtdf = format_df(df,params)
            efmtdf = events_df(fmtdf)
            efmtdf.insert(1,'filename_id',filename_id)
            write_out(df=efmtdf,out=out,type='tsv')

            # make some score output
            if score:
                combined_scores = pd.concat([combined_scores,score_df(efmtdf)],axis=0,ignore_index=True)
            if cov:
                combined_cov = pd.concat([combined_cov,cov_df(efmtdf,window_duration=window)],axis=0,ignore_index=True)

        except Exception as e:
            with open(error_log, 'a') as f:
                f.write(f'{filename} : {e}\n{traceback.format_exc()}\n')
            continue
    
    if score:
        combined_scores.to_csv(os.path.join(out,f"scores_{combined_scores['exp_name'].head(1).values[0]}_n{len(combined_scores)}.csv"),index=False)
    if cov:
        combined_cov.to_csv(os.path.join(out,f"cov_{combined_cov['exp_name'].head(1).values[0]}_{window}s_n{combined_cov['filename_id'].nunique()}.csv"),index=False)

def write_out(df:pd.DataFrame,out:str,type:str,tag:str=''):

    if type == 'csv':
        sep = ','
    elif type == 'tsv':
        sep = '\t'

    df.to_csv(os.path.join(out,f"{df['filename_id'].values[0]}_{df['session'].values[0]}_PGNG{tag}_{df['datetime'].values[0]}.{type}"),index=False,sep=sep)

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

            if params['metacols']['exp_start']:
                tmpdf['exp_start'] = df[params['metacols']['exp_start']].dropna().values[0]

            tmpdf['block'] = block

        except Exception as e:
            with open(error_log, 'a') as f:
                f.write(f"{tmpdf['id'].values[0]} : {e}\n{traceback.format_exc()}\n")
            continue


        fmtdf = pd.concat([fmtdf,tmpdf],ignore_index=True)


    return fmtdf
    
def events_df(df:pd.DataFrame) -> pd.DataFrame:
    '''
    label rows as PGNGS event types
    takes in a formatted dataset
    '''

    dfl = df.set_index('block').groupby(level='block',as_index=False).apply(event_block).reset_index(drop=True)
    dflt = onsets(dfl)
    dflt.insert(dflt.columns.get_loc('trial'),'block',dflt.pop('block'))
    #dflt.drop(columns=dflt.filter(regex='level_.*',axis=1).columns.to_list(),axis=1,inplace=True)

    return dflt

def event_block(block:pd.DataFrame) -> pd.DataFrame:
    '''
    '''

    block.reset_index(inplace=True)
    block['stim_class'] = ''
    block['resp_class'] = ''

    if block['type'].values[0] == 'go':
        block['stim_class'] = block.apply(lambda x: 'target' if x['stimuli'] in x['stim_targ_names'] \
            else '', axis=1)
        
        block['resp_class'] = resp_go(block)

    elif block['type'].values[0] == 'gng':
        block['stim_class'] = stim_gng(block)

        block['resp_class'] = resp_gng(block)
    
    # elif grp['type'].values[0] == 'gs':  # TODO: add stop var
    #     grp['stim_class'] = grp.apply(lambda x: 'lure' if x['stimuli'].shift(1) == "Stop.bmp" \
    #         else 'target' if x['stimuli'] in x['stim_targ_names'] else 'nontarget', axis=1)

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
    #TODO
    pass

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

def resp_gs(grp:pd.DataFrame) -> pd.Series:
    #TODO
    pass

def onsets(df:pd.DataFrame) -> pd.DataFrame:
    #TODO
    '''
    hits rej com om mo
    '''
    df['stim_start_adj'] = np.nan
    df['onsets'] = np.nan
    df['rt_adj'] = np.nan

    for i, row in df.iterrows():
        if df.loc[i,'stim_start'] != '':
            df.loc[i,'stim_start_adj'] = df.loc[i,'stim_start'] - df.loc[i,'exp_start']
        if df.loc[i,'resp_class'] != '':
            if not np.isnan(df.loc[i,'rt']):
                df.loc[i,'onsets'] = df.loc[i,'rt'] + (df.loc[i,'stim_start'] - df.loc[i,'exp_start'])
                df.loc[i,'rt_adj'] = df.loc[i,'rt']
            elif df.loc[i,'stim_class'] != '' and not np.isnan(df.loc[i+1,'rt']):
                df.loc[i,'onsets'] = df.loc[i+1,'rt'] + (df.loc[i+1,'stim_start'] - df.loc[i+1,'exp_start'])
                df.loc[i,'rt_adj'] = df.loc[i+1,'rt'] + (df.loc[i+1,'stim_start'] - df.loc[i,'stim_start'])
            else:
                df.loc[i,'onsets'] = df.loc[i,'stim_start'] - df.loc[i,'exp_start']
        else:
            df.loc[i,'onsets'] = np.nan

    return df

def score_df(df:pd.DataFrame) -> pd.DataFrame:
    '''
    '''
    # setup meta columns
    df_scores = pd.DataFrame({
            'filename_id':df['filename_id'].head(1).values[0],
            'id':df['id'].head(1).values[0],
            'session':df['session'].head(1).values[0],
            'datetime':df['datetime'].head(1).values[0],
            'exp_name':df['exp_name'].head(1).values[0],
            'software_version':df['software_version'].head(1).values[0],
            'framerate':df['framerate'].head(1).values[0]
            },
            index=[0]).reset_index(drop=True)
    
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
            f'gs_{stim_count}T_stp_tm':'', #TODO
            f'gs_{stim_count}T_stp_tm_f':'',
            f'gs_{stim_count}T_pctt':len(block.loc[block['resp_class']=='hit']) / len(block.loc[block['stim_class']=='target']),
            f'gs_{stim_count}T_pcit':''
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
                'filename_id':window['filename_id'].head(1).values[0],
                'id':window['id'].head(1).values[0],
                'session':window['session'].head(1).values[0],
                'datetime':window['datetime'].head(1).values[0],
                'exp_name':window['exp_name'].head(1).values[0],
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

    # for resp_class_str in ['hit','com']:

    #     df_filtered = df.loc[df['resp_class']==resp_class_str]

    #     window_start = df_filtered['onsets'].head(1).values[0]
    #     last_onset = df_filtered['onsets'].tail(1).values[0]

    #     global_window = df_filtered['rt_adj'].values

    #     df_cov = pd.DataFrame({
    #         'filename_id':df_filtered['filename_id'].head(1).values[0],
    #         'id':df_filtered['id'].head(1).values[0],
    #         'session':df_filtered['session'].head(1).values[0],
    #         'datetime':df_filtered['datetime'].head(1).values[0],
    #         'block':np.nan,
    #         'type':'',
    #         'resp_class':'',
    #         'window_start':window_start,
    #         'window_end':last_onset,
    #         'rt_count':len(global_window),
    #         'rt_mean':np.mean(global_window),
    #         'rt_sd':np.std(global_window),
    #         'rt_cov':np.std(global_window)/np.mean(global_window)},
    #         index=[0])
        
    #     df_cov_blocks = df_filtered.set_index('block').groupby('block',as_index=False).apply(lambda x: cov_group(x,window_increment,rt_count))
    #     df_cov = pd.concat([df_cov,df_cov_blocks])
    #     write_out(df=df_cov,out=out,type='csv',tag=f'_COV_{resp_class_str}')
    #     #df_cov.to_csv(os.path.join(out,f"{df['filename_id'].head(1).values[0]}_{df['session'].head(1).values[0]}_PGNG_COV_{resp_class_str}_{df['datetime'].head(1).values[0]}.csv"),index=False)

# def cov_group(df:pd.DataFrame,cov_window_increment:float,cov_rt_count:int) -> pd.DataFrame:

#     df.reset_index(inplace=True)

#     window_start = df['onsets'].head(1).values[0]
#     last_onset = df['onsets'].tail(1).values[0]

#     group_window = df['rt_adj'].values

#     df_cov = pd.DataFrame({'window_start':window_start,'window_end':last_onset,
#                             'rt_count':len(group_window),'rt_mean':np.mean(group_window),
#                             'rt_sd':np.std(group_window),'rt_cov':np.std(group_window)/np.mean(group_window)},
#                             index=[0])

#     while window_start < last_onset:
#         local_window = df[df['onsets'] >= window_start][['onsets','rt_adj']].head(cov_rt_count)

#         if len(local_window['rt_adj']) < cov_rt_count:
#             break

#         window_row = pd.DataFrame({'window_start':window_start,'window_end':local_window['onsets'].tail(1).values[0],
#                                    'rt_count':len(local_window['rt_adj']),'rt_mean':np.mean(local_window['rt_adj']),
#                                    'rt_sd':np.std(local_window['rt_adj']),'rt_cov':np.std(local_window['rt_adj'])/np.mean(local_window['rt_adj'])},
#                                    index=[0])
        
#         df_cov = pd.concat([df_cov,window_row],ignore_index=True,axis=0)

#         window_start += cov_window_increment

#     for var in ['resp_class','type','block','datetime','session','filename_id','id']:
#         df_cov.insert(loc=0, column=var, value=df[var].head(1).values[0])

#     return df_cov
