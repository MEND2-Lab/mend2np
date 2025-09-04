'''

required minimum format:
block_type_col
stimuli_col
response_col

data will be scored grouped by block number

'''

from context import scorenp
from scorenp import pgng

# params structure if data is unformatted:
params = {
    'metacols':{
        'id':'participant',
        'session':'session',
        'datetime':'date',
        'exp_name':'expName',
        'software_version':'psychopyVersion',
        'framerate':'frameRate',
        'exp_start':'expStart'
    },
    'blocks':{
        '1':{
            'metavars':{
                'stim_targ_names':['L_s.bmp','L_r.bmp'],
                'resp_key':'n',
                'stim_dur':0.75,
                'type':'go'
                #'stop':'Stop.bmp'
            },
            'cols':{
                'stimuli':'stimuli_1',
                'stim_start':'block1_stim.started',
                'response':'block1_resp.keys',
                'rt':'block1_resp.rt',
                'trial':'PGNGS_B1.thisTrialN'
            }
        },
        '2':{
            'metavars':{
                'stim_targ_names':['L_s.bmp','L_r.bmp'],
                'resp_key':'n',
                'stim_dur':0.75,
                'type':'gng'
            },
            'cols':{
                'stimuli':'stimuli_2',
                'stim_start':'block2_stim.started',
                'response':'block2_resp.keys',
                'rt':'block2_resp.rt',
                'trial':'PGNGS_B2.thisTrialN'
            }
        },
        '3':{
            'metavars':{
                'stim_targ_names':['L_s.bmp','L_r.bmp','L_t.bmp'],
                'resp_key':'n',
                'stim_dur':0.75,
                'type':'go'
            },
            'cols':{
                'stimuli':'stimuli_4',
                'stim_start':'block4_stim.started',
                'response':'block4_resp.keys',
                'rt':'block4_resp.rt',
                'trial':'PGNGS_B3.thisTrialN'
            }
        },
        '4':{
            'metavars':{
                'stim_targ_names':['L_s.bmp','L_r.bmp','L_t.bmp'],
                'resp_key':'n',
                'stim_dur':0.75,
                'type':'gng'
            },
            'cols':{
                'stimuli':'stimuli_5',
                'stim_start':'block5_stim.started',
                'response':'block5_resp.keys',
                'rt':'block5_resp.rt',
                'trial':'PGNGS_B4.thisTrialN'
            }
        }
    }
}

pgng.main(params=params,formatted=False,out="out")

