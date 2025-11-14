# mend2_neuropsych

Modular scripts in Python to format, score, and extract timing/event information from MEND2 lab neuropsychological tests.

Recommend Python >= 3.12

Install with:
`pip install git+https://github.com/MEND2-Lab/mend2np.git`

Each scoring module contains a main function (e.g., [pgng](mend2np/pgng.py)) to which is passed a list of data filenames, dictionary of parameters, and other configurable options. Example usage is found in the [tests](tests) directory.

#TODO