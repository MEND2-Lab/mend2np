'''
'''

import os
import re
import sys
import logging
import traceback
import pandas as pd
from datetime import datetime
from tkinter import filedialog as fd
from ast import literal_eval

def setup_logger(name:str='root', out:str='out', level:int|str=20):
    os.makedirs(out, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Clear any handlers from a previous call so repeated invocations in one Python
    # session (e.g. running two task modules back-to-back) don't multiplex log lines.
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
    datetime_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(module)s : %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(out,f'log_{datetime_string}.log'),mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def select_files() -> tuple:
    filepaths = fd.askopenfilenames(
        title='Select CSV files to score',
        filetypes=(("CSV Files", "*.csv"),),
        initialdir=os.getcwd(),
        multiple=True)
    return filepaths

def parse_files(filepath:str) -> str:
    # Extract the participant ID prefix from a filename like "<id>_<rest>.csv".
    basename = os.path.basename(filepath)
    match = re.match(r'^[^_]+', basename)
    return match.group(0) if match else ''

def write_out(df:pd.DataFrame,out:str,merged:bool,filetype:str,tag:str='',exp_name:str=''):

    if filetype == 'csv':
        sep = ','
    elif filetype == 'tsv':
        sep = '\t'

    if merged:

        if len(exp_name) == 0 and 'exp_name' in df.columns:
            exp_name = str(df['exp_name'].head(1).values[0]).replace(os.sep,'')

        n_ids = df['id'].nunique() if 'id' in df.columns else len(df)
        filename = f"{exp_name}_n{n_ids}_{tag}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{filetype}"
        df.to_csv(os.path.join(out,filename),index=False,sep=sep)

    else:
        filename = ''
        for var in ['filename_id','id','session','exp_name','datetime']:
            if var in df.columns:
                if not filename:
                    filename =  ''.join([filename,str(df[var].head(1).values[0]).replace(os.sep,'')])
                else:
                    filename = '_'.join([filename,str(df[var].head(1).values[0]).replace(os.sep,'')])
        filename = filename + f'.{filetype}'
        df.to_csv(os.path.join(out,filename),index=False,sep=sep)

def handle_multiple_responses(value, slice_index=0) -> str|float|int|None:
    # If a string representation of a list, and list is not empty, keep only the requested slice.
    # Also defensively handle common non-literal tokens like nan/NaN/None/inf inside the list.

    if isinstance(value, str) and re.match(r'^\[.*\]$', value.strip()):
        s = value.strip()

        # Replace tokens that aren't valid Python literals for literal_eval.
        # (Keep this conservative: only do it for common float-ish NaN/None/inf spellings.)
        s = re.sub(r'\bnan\b', 'None', s, flags=re.IGNORECASE)
        s = re.sub(r'\bNaN\b', 'None', s)
        s = re.sub(r'\binf\b', 'None', s, flags=re.IGNORECASE)
        s = re.sub(r'\b-?inf\b', 'None', s, flags=re.IGNORECASE)
        # JSON null inside list-like strings
        s = re.sub(r'\bnull\b', 'None', s, flags=re.IGNORECASE)
        s = re.sub(r'\bNone\b', 'None', s, flags=re.IGNORECASE)


        try:
            eval_value = literal_eval(s)
        except (ValueError, SyntaxError):
            return value

        if isinstance(eval_value, list):
            if len(eval_value) > 0:
                return eval_value[slice_index]
            # Empty list: return it as a list rather than the original "[]" string,
            # so downstream callers can treat it uniformly with the non-empty case.
            return eval_value

    return value

class ConfigError(Exception):
    """Raised when a task's params dict is missing a required section or has the wrong shape."""


def _resolve_filelist(filelist, logger) -> list:
    """Turn the heterogeneous `filelist` argument into a concrete list of filepaths.

    Accepts a list, a path to a text file with one path per line, or an empty
    value (which triggers the GUI file picker). Exits the process on a
    malformed argument since there is nothing the caller can usefully do.
    """
    if filelist:
        if isinstance(filelist, list):
            return list(filelist)
        if os.path.isfile(filelist):
            try:
                return [line.strip() for line in open(filelist, 'r', encoding='utf-8')]
            except Exception as e:
                logger.critical(f'problem reading filelist: {filelist}: {e}\n{traceback.format_exc()}\n')
                sys.exit(1)
        logger.critical(f'problem with filelist: {filelist}, consult docs or leave blank to use GUI file select')
        sys.exit(1)
    return list(select_files())


def run_task(*, params, filelist, out, write, log, process_file_fn,
             write_trials:bool=True, write_scores:bool=True) -> tuple:
    """Generic per-file processing loop shared by every task module.

    `process_file_fn(filepath, params, logger)` is called once per input file and
    should return a tuple `(trial_df, scores_row)`. Either may be None or empty
    if a task doesn't produce that output type (e.g. fept has no per-trial output).
    Exceptions raised inside the callable are logged and the file is skipped.

    Returns `(combined_scores, combined_trials)`. The caller is responsible for
    setting up the logger (so log-level / module-name choices remain task-local).
    """
    os.makedirs(out, exist_ok=True)
    logger = logging.getLogger('root')
    filepaths = _resolve_filelist(filelist, logger)

    combined_trials = pd.DataFrame()
    combined_scores = pd.DataFrame()
    n_ok = n_skipped = 0

    for filepath in filepaths:
        logger.info(f'processing: {filepath}')
        filename = os.path.basename(filepath)
        try:
            trial_df, scores_row = process_file_fn(filepath, params, logger)
            if trial_df is not None and not trial_df.empty:
                combined_trials = pd.concat([combined_trials, trial_df], axis=0, ignore_index=True)
            if scores_row is not None and not scores_row.empty:
                combined_scores = pd.concat([combined_scores, scores_row], axis=0, ignore_index=True)
            n_ok += 1
        except Exception as e:
            logger.error(f'{filename} : {e}\n{traceback.format_exc()}\n')
            n_skipped += 1

    if write:
        if write_trials and not combined_trials.empty:
            write_out(combined_trials, out, True, 'csv', 'trials')
        if write_scores and not combined_scores.empty:
            write_out(combined_scores, out, True, 'csv', 'scores')

    logger.info(f'scored {n_ok}/{len(filepaths)} files; skipped {n_skipped}')
    logger.info('end')

    return combined_scores, combined_trials


def validate_params(params, schema, prefix:str='params') -> None:
    """Check that `params` matches `schema`, raising ConfigError with a clear message if not.

    schema is a dict whose values are either:
      - a Python type (e.g. dict, str) — checked against isinstance()
      - a nested dict schema — recursively validated against the same key in params
      - None — treat as "must be present, no type check"
    """
    if not isinstance(params, dict):
        raise ConfigError(f"{prefix} must be a dict, got {type(params).__name__}")
    for key, expected in schema.items():
        if key not in params:
            raise ConfigError(
                f"{prefix} is missing required key '{key}'. Check your JSON config — "
                f"compare it against one of the *_example.json files in tests/."
            )
        if isinstance(expected, dict):
            validate_params(params[key], expected, prefix=f"{prefix}.{key}")
        elif isinstance(expected, type):
            if not isinstance(params[key], expected):
                raise ConfigError(
                    f"{prefix}.{key} expected type {expected.__name__}, "
                    f"got {type(params[key]).__name__}"
                )


def get_meta_cols(df,params):
    '''
    for aggregated values (one row per participant) collect meta variables into one row.
    Keys starting with `_` (e.g. `_comment` annotations in JSON configs) are skipped.
    '''

    metacols_df = pd.DataFrame(index=[0])

    for metacol in params['metacols']:
        if metacol.startswith('_'):
            continue
        if params['metacols'][metacol] and metacol in df.columns:
            metacols_df[metacol] = df[metacol].head(1).values[0]

    return metacols_df.reset_index(drop=True)