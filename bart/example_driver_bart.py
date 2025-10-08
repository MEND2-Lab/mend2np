'''
'''

import bart

params = {
    'metacols':{
        'id':'participant',
        'session':'session',
        'datetime':'date',
        'exp_name':'expName',
        'software_version':'psychopyVersion',
        'framerate':'frameRate'
    },
    'cols':{
        'nPumps':'nPumps',
        'popped':'popped',
        'earnings':'earnings',
        'trial':'real_trials.thisN'
    }
}

bart.bart(params=params,out='bart/out')