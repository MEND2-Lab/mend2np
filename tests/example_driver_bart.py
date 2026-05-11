from pathlib import Path

from mend2np.bart import bart

HERE = Path(__file__).parent
data_dir = HERE / 'example_data'
out_dir = HERE / 'out'

filelist = [
    str(data_dir / 'example_data_psychopy_bart_1.csv'),
]

params = {
    'metacols': {
        'id': 'participant',
        'session': 'session',
        'datetime': 'date',
        'exp_name': 'expName',
        'software_version': 'psychopyVersion',
        'framerate': 'frameRate',
        'os': 'OS'
    },
    'cols': {
        'nPumps': 'nPumps',
        'popped': 'popped',
        'earnings': 'earnings',
        'trial': 'real_trials.thisN',
        'rt': 'mouse.time'
    }
}

bart(params=params, out=str(out_dir), filelist=filelist)
