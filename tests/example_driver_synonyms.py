
from mend2np.synonyms import synonyms
import json

with open('tests/synonyms_example.json','r') as f:
    params = json.load(f)
    synonyms(params=params,formatted=False,out="tests/out")