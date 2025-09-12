'''



'''

from context import scorenp
from scorenp import pgng

# params structure if data is unformatted:
params = {
    'metacols':{
        'id':'participant',  # required
        'session':'session',
        'datetime':'date',
        'exp_name':'expName',
        'software_version':'psychopyVersion',
        'framerate':'frameRate',
        'exp_start':'expStart'  # required for timing
    },
    'blocks':{
        '1':{
            'metavars':{
                'stim_targ_names':['L_s.bmp','L_r.bmp'],  # required
                'resp_key':'n',  # required
                'stim_dur':0.75,  # required for timing
                'type':'go'  # required
                #'stop':'Stop.bmp'
            },
            'cols':{
                'stimuli':'stimuli_1',  # required
                'stim_start':'block1_stim.started',  # required for timing
                'response':'block1_resp.keys',  # required
                'rt':'block1_resp.rt',  # required
                'trial':'PGNGS_B1.thisTrialN'  # required
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

#pgng.main(params=params,formatted=False,out="out",filelist=['tests/example_data/example_data_psychopy_pgng_1.csv'])

pgng.pgng(params=params,formatted=False,out="out",cov_window=50)
