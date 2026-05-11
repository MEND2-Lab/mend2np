from pathlib import Path
import json

from mend2np.pgng import pgng

HERE = Path(__file__).parent
data_dir = HERE / 'example_data'
out_dir = HERE / 'out'

filelist = [
    str(data_dir / 'example_data_psychopy_pgng_2.csv'),
    str(data_dir / 'example_data_psychopy_pgng_3.csv'),
]

with open(HERE / 'pgng_example.json', 'r') as f:
    params = json.load(f)

pgng(params=params, formatted=False, out=str(out_dir), filelist=filelist)
