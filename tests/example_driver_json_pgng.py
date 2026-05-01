'''

'''
from mend2np.pgng import pgng
import json

with open('tests/pgng_example.json','r') as f:
    params = json.load(f)
    pgng(params=params,formatted=False,out="tests/out")