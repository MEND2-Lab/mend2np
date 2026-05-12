"""Driver script for Finger Oscillation scoring.

The bundled example CSVs come from four different versions of the task with
three different column-layout shapes. Running this script scores all four,
writing one set of `_trials_<timestamp>.csv` / `_scores_<timestamp>.csv` per
variant. The output filename's `exp_name` prefix tells you which run is which.
"""

from pathlib import Path
import json

from mend2np.fingosc import fingosc

HERE = Path(__file__).parent
data_dir = HERE / 'example_data'
out_dir = HERE / 'out'


def load(name):
    with open(HERE / name, 'r') as f:
        return json.load(f)


# test2 / test3 — keyboard, per-block columns (dominant_key_resp / nondominant_key_resp).
fingosc(
    params=load('fingosc_example.json'),
    formatted=False,
    out=str(out_dir),
    filelist=[
        str(data_dir / 'test2_fo.csv'),
        str(data_dir / 'test3_fo.csv'),
    ],
)

# test1 — touch, per-block columns (mouse_Right / mouse_Left).
fingosc(
    params=load('fingosc_example_touch.json'),
    formatted=False,
    out=str(out_dir),
    filelist=[
        str(data_dir / 'test1_fo.csv'),
    ],
)

# test4 — single column set; block-end marker rows distinguish the two blocks.
fingosc(
    params=load('fingosc_example_stacked.json'),
    formatted=False,
    out=str(out_dir),
    filelist=[
        str(data_dir / 'test4_fo.csv'),
    ],
)
