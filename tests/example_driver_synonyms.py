from pathlib import Path
import json

from mend2np.synonyms import synonyms

HERE = Path(__file__).parent
data_dir = HERE / 'example_data'
out_dir = HERE / 'out'

filelist = [
    str(data_dir / 'example_data_synonyms_1.csv'),
]

with open(HERE / 'synonyms_example.json', 'r') as f:
    params = json.load(f)

synonyms(params=params, formatted=False, out=str(out_dir), filelist=filelist)
