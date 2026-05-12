"""Driver script for Stroop scoring.

Three example CSVs: test1 & test2 are keyboard-response runs (use the default
config); test3 is a touch-response run (use the touch config — same column
mappings except for the response/rt columns).
"""

from pathlib import Path
import json

from mend2np.stroop import stroop

HERE = Path(__file__).parent
data_dir = HERE / 'example_data'
out_dir = HERE / 'out'


def load(name):
    with open(HERE / name, 'r') as f:
        return json.load(f)


# Keyboard runs.
stroop(
    params=load('stroop_example.json'),
    formatted=False,
    out=str(out_dir),
    filelist=[
        str(data_dir / 'test1_stroop.csv'),
        str(data_dir / 'test2_stroop.csv'),
    ],
)

# Touch run.
stroop(
    params=load('stroop_example_touch.json'),
    formatted=False,
    out=str(out_dir),
    filelist=[
        str(data_dir / 'test3_stroop.csv'),
    ],
)
