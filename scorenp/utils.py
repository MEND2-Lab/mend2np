import os
import re
from datetime import datetime
from tkinter import filedialog as fd

# def main(task:str, platform:str='', filepaths:list=[], out:str='./', params:dict={}):
#     print(task)
#     print(platform)

#     # temporary error log, add proper logging later
#     error_log = Path(out,'error_log.csv')
#     if os.path.exists(error_log):
#         os.remove(error_log)

#     #merged_data = pd.DataFrame()

#     #row = 0
#     for filepath in select_files():

#         id,dt,basename = parse_files(filepath)
#         id_dict = {
#             'id':id,
#             'dt':dt,
#             'filename':basename
#         }

#         filepath = Path(filepath)
#         # basic quality checks
#         try:
#             pd.read_csv(filepath)
#         except Exception as e:
#             with open(error_log, 'a') as f:
#                 f.write(f'{basename} : {e}\n')
#             continue

#         if task == 'pgng':
#             scores,onsets = pgng(filepath)
#         else:
#             print('task not recognized')

#         merged_dicts = id_dict | scores | onsets

#         # concat with pd dataframe

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
            #dt = datetime.strptime(date_str,"%Y-%m-%d_%Hh%M.%S.%f")
            dt = datetime.strptime(date_str,fmt)
            break
        except ValueError:
            pass
    id = re.match(r'^\d+',basename).group(0)

    return (id,dt,basename)