'''

'id'                required
'filename_id'       not required
'session'           not required
'datetime'          recommended, not required
'exp_name'          not required
'software_version'  not required
'framerate'         not required
'exp_start'         required for timing
'stimuli'           required
'stim_start'        required for timing
'response'          required
'rt'                required
'trial'             not required
'block'             required
'stim_targ_names'   required
'resp_key'          required
'stim_dur'          required for timing
'type'              required

required: id, stimuli, response, rt, block, stim_targ_names, resp_key, type
addtional required for timing: exp_start, stim_start, stim_dur

'''

from context import scorenp
from scorenp import pgng

# params structure if data is formatted:
params = {
    'cols':{
        'id':'id',
        'filename_id':'filename_id',
        'session':'session',
        'datetime':'datetime',
        'exp_name':'expName',
        'software_version':'software_version',
        'framerate':'framerate',
        'exp_start':'expStart',
        'stimuli':'stimuli',
        'stim_start':'stim_start',
        'response':'response',
        'rt':'rt',
        'trial':'trial',
        'block':'block',
        'stim_targ_names':'stim_targ_names',
        'resp_key':'resp_key',
        'stim_dur':'stim_dur',
        'type':'type'
    }
}

pgng.main(params=params,formatted=True,cov=True,out="out",filelist="filepaths.txt")
