'''

'''
import pandas as pd
from . import utils
import os
import numpy as np
import traceback

def main(params:dict,formatted:bool=False,out:str=os.getcwd()):
    '''
    '''

    os.makedirs(out, exist_ok=True)

    # temporary error log, add proper logging later
    error_log = os.path.join(out,'error_log.csv')
    if os.path.exists(error_log):
        os.remove(error_log)

    for filepath in utils.select_files():
        #idx,dt,basename = utils.parse_files(filepath)
        filename = os.path.basename(filepath)

        try:
            df = pd.read_csv(filepath)
            fmtdf = format(df,params)
            efmtdf = events(fmtdf)
            cov(efmtdf,out)
        except Exception as e:
            with open(error_log, 'a') as f:
                f.write(f'{filename} : {e}\n{traceback.format_exc()}\n')
            continue

        # fmtdf = format(df,params)
        # efmtdf = events(fmtdf)
        # cov(efmtdf,out)

        efmtdf.to_csv(os.path.join(out,f"{efmtdf['id'].values[0]}_{efmtdf['session'].values[0]}_PGNG_{efmtdf['datetime'].values[0]}.tsv"),index=False,sep='\t')

def format(df:pd.DataFrame,params:dict) -> pd.DataFrame:
    '''
    '''

    fmtdf = pd.DataFrame()

    for block in params['blocks']:

        tmpdf = pd.DataFrame()
        
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

        fmtdf = pd.concat([fmtdf,tmpdf],ignore_index=True)


    return fmtdf
    
def events(df:pd.DataFrame) -> pd.DataFrame:
    '''
    label rows as PGNGS event types
    takes in a formatted dataset
    '''

    dfl = df.set_index('block').groupby(level='block',as_index=False).apply(event_lbl).reset_index()
    dflt = onsets(dfl)
    dflt.insert(dflt.columns.get_loc('trial'),'block',dflt.pop('block'))
    dflt.drop(columns=dflt.filter(regex='level_.*',axis=1).columns.to_list(),axis=1,inplace=True)

    return dflt

def event_lbl(grp:pd.DataFrame) -> pd.DataFrame:
    '''
    '''

    grp.reset_index(inplace=True)
    grp['stim_class'] = ''
    grp['resp_class'] = ''

    if grp['type'].values[0] == 'go':
        grp['stim_class'] = grp.apply(lambda x: 'target' if x['stimuli'] in x['stim_targ_names'] \
            else '', axis=1)
        
        grp['resp_class'] = resp_go(grp)

    elif grp['type'].values[0] == 'gng':
        grp['stim_class'] = stim_gng(grp)

        grp['resp_class'] = resp_gng(grp)
    
    # elif grp['type'].values[0] == 'gs':  # TODO: add stop var
    #     grp['stim_class'] = grp.apply(lambda x: 'lure' if x['stimuli'].shift(1) == "Stop.bmp" \
    #         else 'target' if x['stimuli'] in x['stim_targ_names'] else 'nontarget', axis=1)

    return grp

def stim_gng(grp:pd.DataFrame) -> pd.Series:
    '''
    '''

    targs = grp['stim_targ_names'].values[0]

    last_seen = dict(zip(targs, [None] * len(targs)))

    for i, current in grp['stimuli'].items():
        if current in last_seen:
            grp.at[i,'stim_class'] = 'target' if last_seen[current] is None else 'lure'
            last_seen = {key: None for key, value in last_seen.items()}
            last_seen[current] = i
        else:
            grp.at[i,'stim_class'] = ''
    
    return grp['stim_class']

def stim_gs(grp:pd.DataFrame) -> pd.Series:
    #TODO
    pass

def resp_go(grp:pd.DataFrame) -> pd.Series:
    '''
    '''

    for i, row in grp.iterrows():
        if grp.loc[i,'stim_class'] == 'target':
            if grp.loc[i,'response'] == grp.loc[i,'resp_key'] or grp.loc[i+1,'response'] == grp.loc[i+1,'resp_key']:
                grp.loc[i,'resp_class'] = 'hit'
            else:
                grp.loc[i,'resp_class'] = 'om'
        elif i>0 and grp.loc[i-1,'stim_class'] != 'target' and grp.loc[i,'response'] == grp.loc[i,'resp_key']:
            grp.loc[i,'resp_class'] = 'randcom'
        else:
            grp.loc[i,'resp_class'] = ''

    return grp['resp_class']

def resp_gng(grp:pd.DataFrame) -> pd.Series:
    '''
    '''
    missed = False
    for i, row in grp.iterrows():
        if grp.loc[i,'stim_class'] == 'target':
            if grp.loc[i,'response'] == grp.loc[i,'resp_key'] or grp.loc[i+1,'response'] == grp.loc[i+1,'resp_key']:
                grp.loc[i,'resp_class'] = 'hit'
                missed = False
            else:
                grp.loc[i,'resp_class'] = 'om'
                missed = True
        elif grp.loc[i,'stim_class'] == 'lure':
            if missed:
                grp.loc[i,'resp_class'] = 'mo'
            elif grp.loc[i,'response'] == grp.loc[i,'resp_key'] or grp.loc[i+1,'response'] == grp.loc[i+1,'resp_key']:
                grp.loc[i,'resp_class'] = 'com'
            else:
                grp.loc[i,'resp_class'] = 'rej'
        elif i>0 and grp.loc[i-1,'stim_class'] not in ['target','lure'] and grp.loc[i,'response'] == grp.loc[i,'resp_key']:
            grp.loc[i,'resp_class'] = 'randcom'
        else:
            grp.loc[i,'resp_class'] = ''
            
    return grp['resp_class']

def resp_gs(grp:pd.DataFrame) -> pd.Series:
    #TODO
    pass

def onsets(df:pd.DataFrame) -> pd.DataFrame:
    #TODO
    '''
    hits rej com om mo
    '''

    df['onsets'] = np.nan
    df['rt_adj'] = np.nan

    for i, row in df.iterrows():
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

def score():
    #TODO
    pass

def cov(df:pd.DataFrame,out:str):
    '''
    coefficient of variation
    '''
    window_increment = 7.5
    rt_count = 12

    for resp_class_str in ['hit','com']:

        df_filtered = df.loc[df['resp_class']==resp_class_str]

        window_start = df_filtered['onsets'].head(1).values[0]
        last_onset = df_filtered['onsets'].tail(1).values[0]

        global_window = df_filtered['rt_adj'].values

        df_cov = pd.DataFrame({'id':df_filtered['id'].head(1).values[0],
                            'session':df_filtered['session'].head(1).values[0],
                            'datetime':df_filtered['datetime'].head(1).values[0],
                            'block':np.nan,
                            'type':'',
                            'resp_class':'',
                            'window_start':window_start,
                            'window_end':last_onset,
                            'rt_count':len(global_window),
                            'rt_mean':np.mean(global_window),
                            'rt_sd':np.std(global_window),
                            'rt_cov':np.std(global_window)/np.mean(global_window)},
                                index=[0])
        
        df_cov_blocks = df_filtered.set_index('block').groupby('block',as_index=False).apply(lambda x: cov_group(x,window_increment,rt_count))
        df_cov = pd.concat([df_cov,df_cov_blocks])
        df_cov.to_csv(os.path.join(out,f"{df['id'].head(1).values[0]}_{df['session'].head(1).values[0]}_COV_{resp_class_str}_{df['datetime'].head(1).values[0]}.csv"),index=False)

def cov_group(df:pd.DataFrame,cov_window_increment:float,cov_rt_count:int) -> pd.DataFrame:

    df.reset_index(inplace=True)

    window_start = df['onsets'].head(1).values[0]
    last_onset = df['onsets'].tail(1).values[0]

    group_window = df['rt_adj'].values

    df_cov = pd.DataFrame({'window_start':window_start,'window_end':last_onset,
                            'rt_count':len(group_window),'rt_mean':np.mean(group_window),
                            'rt_sd':np.std(group_window),'rt_cov':np.std(group_window)/np.mean(group_window)},
                            index=[0])

    while window_start < last_onset:
        local_window = df[df['onsets'] >= window_start][['onsets','rt_adj']].head(cov_rt_count)

        if len(local_window['rt_adj']) < cov_rt_count:
            break

        window_row = pd.DataFrame({'window_start':window_start,'window_end':local_window['onsets'].tail(1).values[0],
                                   'rt_count':len(local_window['rt_adj']),'rt_mean':np.mean(local_window['rt_adj']),
                                   'rt_sd':np.std(local_window['rt_adj']),'rt_cov':np.std(local_window['rt_adj'])/np.mean(local_window['rt_adj'])},
                                   index=[0])
        
        df_cov = pd.concat([df_cov,window_row],ignore_index=True,axis=0)

        window_start += cov_window_increment

    for var in ['resp_class','type','block','datetime','session','id']:
        df_cov.insert(loc=0, column=var, value=df[var].head(1).values[0])

    return df_cov
