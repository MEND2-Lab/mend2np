"""mend2np — modular scoring for MEND2 lab neuropsych task data.

Each task is exposed as a top-level function:
    from mend2np import sert, pgng, bart, fept, synonyms, fingosc, smid, stroop

Debugging helpers:
    from mend2np import preflight_check   # validate a config + CSVs before scoring

See `tests/example_driver_*.py` for usage examples.
"""

from mend2np.sert import sert
from mend2np.pgng import pgng
from mend2np.bart import bart
from mend2np.fept import fept
from mend2np.synonyms import synonyms
from mend2np.fingosc import fingosc
from mend2np.smid import smid
from mend2np.stroop import stroop
from mend2np.utils import preflight_check, ConfigError

__all__ = ['sert', 'pgng', 'bart', 'fept', 'synonyms', 'fingosc', 'smid', 'stroop',
           'preflight_check', 'ConfigError']
