"""Driver script for BART scoring (JSON-config flavour).

Loads the config from `bart_example.json` rather than hardcoding it in Python —
recommended approach for non-programmer use. Equivalent to
`example_driver_bart.py`; both produce identical output.
"""

from pathlib import Path
import json

from mend2np.bart import bart

HERE = Path(__file__).parent
data_dir = HERE / 'example_data'
out_dir = HERE / 'out'

with open(HERE / 'bart_example.json', 'r') as f:
    params = json.load(f)

filelist = [
    str(data_dir / 'example_data_psychopy_bart_1.csv'),
]

bart(params=params, out=str(out_dir), filelist=filelist)
