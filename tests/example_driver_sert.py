
from mend2np.sert import sert
import json

with open('tests/sert_example.json','r') as f:
    params = json.load(f)
    sert(params=params,formatted=False,out="tests/out")