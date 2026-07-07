from pathlib import Path
import json

from mend2np.sert import sert

HERE = Path(__file__).parent
data_dir = HERE / 'example_data'
out_dir = HERE / 'out'

filelist = [
    str(data_dir / 'example_data_psychopy_sert_1.csv'),
    str(data_dir / 'example_data_psychopy_sert_2.csv'),
    str(data_dir / 'example_data_psychopy_sert_3.csv'),
]

with open(HERE / 'sert_example.json', 'r') as f:
    params = json.load(f)

# The Pavlovia SERT runs fixed switch/repeat blocks, so opt into block-level scoring.
sert(params=params, formatted=False, out=str(out_dir), filelist=filelist, block_switch_rep=True)
