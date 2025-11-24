'''
'''

from mend2np.bart import bart

params = {
    'metacols':{
        'id':'participant',
        'session':'session',
        'datetime':'date',
        'exp_name':'expName',
        'software_version':'psychopyVersion',
        'framerate':'frameRate',
        'os':'OS'
    },
    'cols':{
        'nPumps':'nPumps',
        'popped':'popped',
        'earnings':'earnings',
        'trial':'real_trials.thisN',
        'rt':'mouse.time'
    }
}

bart(params=params,out='tests/out')