"""Driver script for SMID scoring.

The bundled CSVs are 4 keyboard-response runs of the Social MID:
  test1, test2 are short practice-only-ish runs (~5 real trials).
  test3, test4 are full runs (~100 real trials).
The same config handles all four — practice trials use a `p`-prefixed
column set, real trials use the unprefixed equivalent.
"""

from pathlib import Path
import json

from mend2np.smid import smid

HERE = Path(__file__).parent
data_dir = HERE / 'example_data'
out_dir = HERE / 'out'

with open(HERE / 'smid_example.json', 'r') as f:
    params = json.load(f)

filelist = [
    str(data_dir / 'test1_smid.csv'),
    str(data_dir / 'test2_smid.csv'),
    str(data_dir / 'test3_smid.csv'),
    str(data_dir / 'test4_smid.csv'),
]

smid(
    params=params,
    formatted=False,
    out=str(out_dir),
    filelist=filelist,
)
