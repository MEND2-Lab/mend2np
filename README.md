# mend2_neuropsych

Modular scripts in Python to format, score, and extract timing/event information from MEND2 lab neuropsychological tests.

Each test has one Python script (pgng.py) which contains all the relevant functions for that test. The main function (pgng.pgng()) can be called from a separate Python script (example_driver_pgng.py) and passed a dictionary of parameters for your particular version of the test.

Recommend Python 3.13.6

Dependencies are listed in requirements.txt and can be installed into your Python environment with `pip install -r requirements.txt1`

#TODO