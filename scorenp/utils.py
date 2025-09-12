import os
import re
import logging
from datetime import datetime
from tkinter import filedialog as fd

def setup_logger(name,out):
    datetime_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(module)s : %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(out,f'log_{datetime_string}.log'),mode='w')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

def select_files() -> tuple:
    filepaths = fd.askopenfilenames(
        title='Select CSV files to score',
        filetypes=(("CSV Files", "*.csv"),),
        initialdir=os.getcwd(),
        multiple=True)
    return filepaths

def parse_files(filepath:str) -> tuple:
    # parse file name into useful bits
    basename = os.path.basename(filepath)
    base = basename.rsplit('.', 1)[0]  
    parts = base.split('_')
    date_str = parts[-2] + '_' + parts[-1]
    for fmt in ["%Y-%m-%d_%Hh%M.%S.%f","%m-%d-%Y_%Hh%M.%S.%f"]:
        try:
            dt = datetime.strptime(date_str,fmt)
            break
        except ValueError:
            pass
    id = re.match(r'^[^_]+',basename).group(0)

    #return (id,dt,basename)
    return (id)