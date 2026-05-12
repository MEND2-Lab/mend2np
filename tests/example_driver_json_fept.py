"""Driver script for FEPT scoring (JSON-config flavour).

Loads the config from `fept_example.json` rather than hardcoding it in Python —
recommended approach for non-programmer use. Equivalent to
`example_driver_fept.py`; both produce identical output.
"""

from pathlib import Path
import json

from mend2np.fept import fept

HERE = Path(__file__).parent
data_dir = HERE / 'example_data'
out_dir = HERE / 'out'

with open(HERE / 'fept_example.json', 'r') as f:
    params = json.load(f)

filelist = [
    str(data_dir / 'example_data_psychopy_fept_1.csv'),
    str(data_dir / 'example_data_psychopy_fept_2.csv'),
]

fept(params=params, out=str(out_dir), filelist=filelist)
