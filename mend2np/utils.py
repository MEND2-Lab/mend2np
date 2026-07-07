'''
Shared utilities used by every task module.

Contents:
- setup_logger          — configure the package 'mend2np' logger (stream + file handlers)
- select_files          — tkinter GUI fallback when no `filelist` is supplied
- parse_files           — extract the participant-ID prefix from a filename
- write_out             — write a trials or scores dataframe to a timestamped CSV/TSV
- handle_multiple_responses — parse "[a, b, c]" cells back into Python lists
- ConfigError           — raised by validate_params on a malformed config
- _resolve_filelist     — internal: turn the heterogeneous `filelist` arg into a list of paths
- run_task              — the per-file processing loop shared by every task module
- validate_params       — shape-check a params dict against a small schema
- get_meta_cols         — collect per-participant metadata for the scores output
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


def setup_logger(name:str='mend2np', out:str='out', level:int|str=20, logfile:bool=False) -> logging.Logger:
    """Configure (or re-configure) the mend2np logger with a stream (and optional file) handler.

    Configures the package's *own* named logger (``'mend2np'`` by default) rather
    than the root logger, and sets ``propagate=False`` so records never reach the
    root logger. This keeps mend2np from mutating the level or handlers of a host
    application's root logger — calling a scoring function no longer silently
    changes the caller's logging configuration. Every task module logs through a
    ``logging.getLogger(__name__)`` child (e.g. ``'mend2np.bart'``), which inherits
    this logger's level and propagates up to its handlers.

    A console (stream) handler is always attached. A timestamped
    ``log_<timestamp>.log`` file is written to ``out`` only when ``logfile=True``,
    so scoring a batch doesn't drop a log file into the output directory unless
    the caller asks for one.

    The same logger instance is returned on every call — Python's `logging`
    module caches loggers by name. We clear any previously-attached handlers
    first so running two task modules back-to-back in one Python session
    doesn't produce duplicate log lines.

    :param name: logger name to configure; defaults to the package logger ``'mend2np'``.
    :param out: directory where `log_<timestamp>.log` will be written when ``logfile=True``; created if missing.
    :param level: numeric log level (10=DEBUG, 20=INFO, 30=WARNING) or a string equivalent.
    :param logfile: if True, also write a timestamped log file to ``out`` (default False).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Don't leak mend2np's records into the host application's root logger.
    logger.propagate = False
    # Clear any handlers from a previous call so repeated invocations in one Python
    # session (e.g. running two task modules back-to-back) don't multiplex log lines.
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(module)s : %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if logfile:
        os.makedirs(out, exist_ok=True)
        datetime_string = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(os.path.join(out,f'log_{datetime_string}.log'),mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def select_files() -> tuple:
    """Pop up a tkinter file-picker for CSV files; return the user's selections.

    Used as a fallback when no `filelist` argument is passed to a task entry
    function. Returns an empty tuple if the user cancels.
    """
    filepaths = fd.askopenfilenames(
        title='Select CSV files to score',
        filetypes=(("CSV Files", "*.csv"),),
        initialdir=os.getcwd(),
        multiple=True)
    return filepaths


def parse_files(filepath:str) -> str:
    """Extract the participant-ID prefix from a filename of the form `<id>_<rest>.csv`.

    Returns the substring up to (not including) the first underscore in the
    basename, or '' if the basename starts with an underscore.
    """
    basename = os.path.basename(filepath)
    # `^[^_]+` matches one or more non-underscore characters at the start.
    match = re.match(r'^[^_]+', basename)
    return match.group(0) if match else ''


def write_out(df:pd.DataFrame, out:str, merged:bool, filetype:str, tag:str='', exp_name:str=''):
    """Write a dataframe to `out/` with a stable, timestamped filename.

    Two filename conventions:
      - `merged=True`  → `<exp_name>_n<N>_<tag>_<timestamp>.<filetype>`
        Used for the combined trial- or score-level outputs.
      - `merged=False` → `<filename_id>_<id>_<session>_<exp_name>_<datetime>.<filetype>`
        Used when writing a per-file copy (the `ind=True` driver option).

    :param df: dataframe to write.
    :param out: output directory (assumed to already exist).
    :param merged: True for a single combined-batch file, False for per-input-file output.
    :param filetype: `'csv'` or `'tsv'` — determines the separator.
    :param tag: suffix word for the merged-mode filename (typically `'trials'` or `'scores'`).
    :param exp_name: overrides the value read from `df['exp_name']` in merged mode.
    """
    # Map filetype to delimiter. Anything other than csv/tsv falls through with sep undefined.
    if filetype == 'csv':
        sep = ','
    elif filetype == 'tsv':
        sep = '\t'

    if merged:
        # Default the experiment name from the data if the caller didn't override it.
        if len(exp_name) == 0 and 'exp_name' in df.columns:
            exp_name = str(df['exp_name'].head(1).values[0]).replace(os.sep,'')

        # `n<N>` reflects unique participant count; falls back to row count if no id column.
        n_ids = df['id'].nunique() if 'id' in df.columns else len(df)
        filename = f"{exp_name}_n{n_ids}_{tag}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{filetype}"
        df.to_csv(os.path.join(out, filename), index=False, sep=sep)

    else:
        # Per-file mode: stitch together as many of these identifying fields as we have.
        filename = ''
        for var in ['filename_id', 'id', 'session', 'exp_name', 'datetime']:
            if var in df.columns:
                if not filename:
                    filename = ''.join([filename, str(df[var].head(1).values[0]).replace(os.sep,'')])
                else:
                    filename = '_'.join([filename, str(df[var].head(1).values[0]).replace(os.sep,'')])
        filename = filename + f'.{filetype}'
        df.to_csv(os.path.join(out, filename), index=False, sep=sep)


def handle_multiple_responses(value, slice_index=0):
    """Parse a string representation of a list (e.g. `"[1, 2, nan]"`) back into a Python list.

    PsychoPy serializes list-valued columns as their `repr()`. When pandas reads
    the CSV back in, each cell is a string like `"['n']"` or `"[1.0, 2.0]"`.
    `literal_eval` will turn the well-formed ones back into lists, but bare
    `nan`/`NaN`/`inf`/`null`/`None` tokens inside the string aren't valid
    Python literals — those are normalised to `None` first.

    :param value: the cell value as pandas reads it (often a string).
    :param slice_index: `0` to pick the first element, `slice(None)` for the
        whole list, or any other indexable selector.
    :returns: the requested slice of the parsed list, or the original value
        when the cell wasn't a list-string at all.
    """
    if isinstance(value, str) and re.match(r'^\[.*\]$', value.strip()):
        s = value.strip()

        # Replace non-literal "missing" tokens (and any leading `-` sign) with `None`
        # so literal_eval will accept the list. The lookbehind/ahead `(?<!\w)`/`(?!\w)`
        # are word-boundary guards that *also* allow `-` to consume a leading minus
        # (which a plain `\b` doesn't — `\b` only matches between a word and non-word
        # char, and `-` is non-word, so `\b-?inf\b` never actually matched `-inf`).
        s = re.sub(r'(?<!\w)-?(?:nan|inf|null|none)(?!\w)', 'None', s, flags=re.IGNORECASE)

        try:
            eval_value = literal_eval(s)
        except (ValueError, SyntaxError):
            # Not parseable — give the original string back so caller can decide.
            return value

        if isinstance(eval_value, list):
            if len(eval_value) > 0:
                return eval_value[slice_index]
            # Empty list means no response was recorded. When the caller wants a
            # scalar (integer slice_index), return NaN so CSV output stays blank
            # rather than writing the string '[]'. When the caller wants the whole
            # list (slice_index=slice(None)), return [] so they can test len()==0.
            return float('nan') if isinstance(slice_index, int) else eval_value

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


def run_task(*, params, filelist, out, write, process_file_fn,
             write_trials:bool=True, write_scores:bool=True) -> tuple:
    """Generic per-file processing loop shared by every task module.

    `process_file_fn(filepath, params, logger)` is called once per input file and
    should return a tuple `(trial_df, scores_row)`. Either may be None or empty
    if a task doesn't produce that output type (e.g. fept has no per-trial output).
    Exceptions raised inside the callable are logged and the file is skipped.

    Returns `(combined_scores, combined_trials)`. The caller is responsible for
    setting up the logger via `setup_logger` *before* calling this — `run_task`
    just reads the 'mend2np' logger; it doesn't configure log level itself.
    """
    os.makedirs(out, exist_ok=True)
    logger = logging.getLogger(__name__)
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


def copy_configured_columns(fmtdf:pd.DataFrame, df:pd.DataFrame, config_section:dict,
                            section_label:str, mask=None, *, logger=None) -> None:
    """Copy each `{standard_name: csv_col}` pair from `config_section` into `fmtdf`.

    Shared by every task's `format_df`. The behaviour is:

      - Skip entries whose key starts with `_` (so JSON annotations like `_comment`
        don't try to map to a CSV column).
      - Skip entries whose value is empty/None (user explicitly disabled that field).
      - If the configured CSV column name is in `df.columns`, copy it (optionally
        masked) into `fmtdf` under the standard name.
      - Otherwise emit a WARNING that names the config path AND the missing CSV
        column — this is the single biggest debugging aid for "my output is
        missing column X" problems.

    :param fmtdf: target dataframe being built up (modified in place).
    :param df: raw input dataframe.
    :param config_section: subdict mapping standard names → CSV column names.
    :param section_label: dotted path used in the warning message (e.g. `'cols'`,
        `'metacols'`, or `'blocks.1.cols'`).
    :param mask: optional boolean mask applied to `df.loc[mask, csv_col]`.
    :param logger: defaults to the module logger ('mend2np.utils') when omitted.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    for standard_name, csv_col in config_section.items():
        if isinstance(standard_name, str) and standard_name.startswith('_'):
            continue
        if not csv_col:
            continue
        if csv_col in df.columns:
            if mask is not None:
                fmtdf[standard_name] = df.loc[mask, csv_col]
            else:
                fmtdf[standard_name] = df[csv_col]
        else:
            logger.warning(
                f"{section_label}.{standard_name}: configured CSV column '{csv_col}' "
                f"is not in this file's columns — '{standard_name}' will be missing "
                f"from the output. Check the spelling in your config, or set this "
                f"entry's value to '' to skip silently."
            )


def preflight_check(params:dict, filelist:list, task:str, *, verbose:bool=True) -> dict:
    """Validate a config and one or more CSVs without doing any scoring.

    Walks the config (`metacols`, `cols`, and `blocks[N].cols` where applicable),
    enumerates every CSV column name it expects to find, and reports any that
    aren't in each provided CSV. This is the recommended first step when a user
    reports "the output is wrong" — it answers the most common cause (a typo'd
    column name in the JSON) without running any scoring code.

    :param params: the config dict (as loaded from JSON or constructed in Python).
    :param filelist: list of CSV file paths to validate against.
    :param task: 'sert', 'pgng', 'bart', 'fept', or 'synonyms' — informs which
        sections of the config to walk.
    :param verbose: if True (default), print a human-readable report to stdout.
    :returns: dict mapping each CSV path to a list of `(config_path, csv_col)`
        tuples describing every missing reference. An empty list means that CSV is fine.
    """
    valid_tasks = ('sert', 'pgng', 'bart', 'fept', 'synonyms', 'fingosc', 'smid', 'stroop')
    if task not in valid_tasks:
        raise ValueError(f"task must be one of {valid_tasks}, got {task!r}")

    # Enumerate every (config-path, csv-column) reference the task will look for.
    expected_refs:list[tuple[str,str]] = []
    if 'metacols' in params and isinstance(params['metacols'], dict):
        for k, v in params['metacols'].items():
            if isinstance(k, str) and k.startswith('_'):
                continue
            if isinstance(v, str) and v:
                expected_refs.append((f'metacols.{k}', v))
    if 'cols' in params and isinstance(params['cols'], dict):
        for k, v in params['cols'].items():
            if isinstance(k, str) and k.startswith('_'):
                continue
            if isinstance(v, str) and v:
                expected_refs.append((f'cols.{k}', v))
    if task in ('pgng', 'fept', 'fingosc', 'smid') and 'blocks' in params and isinstance(params['blocks'], dict):
        for block_key, block_cfg in params['blocks'].items():
            if isinstance(block_key, str) and block_key.startswith('_'):
                continue
            if not isinstance(block_cfg, dict):
                continue
            block_cols = block_cfg.get('cols', {})
            if not isinstance(block_cols, dict):
                continue
            for k, v in block_cols.items():
                if isinstance(k, str) and k.startswith('_'):
                    continue
                if isinstance(v, str) and v:
                    expected_refs.append((f'blocks.{block_key}.cols.{k}', v))

    issues_by_file:dict[str, list] = {}
    for filepath in filelist:
        try:
            # nrows=0 reads the header only; no need to load the whole file.
            df_cols = pd.read_csv(filepath, nrows=0).columns.tolist()
        except Exception as e:
            issues_by_file[filepath] = [('<read-error>', str(e))]
            continue
        df_cols_set = set(df_cols)
        missing = [(path, csv_col) for path, csv_col in expected_refs if csv_col not in df_cols_set]
        issues_by_file[filepath] = missing

    if verbose:
        print(f"\npreflight_check ({task}) — {len(expected_refs)} configured column references\n")
        for filepath, missing in issues_by_file.items():
            if missing and missing[0][0] == '<read-error>':
                print(f"[ERROR] {filepath}")
                print(f"        could not read CSV: {missing[0][1]}")
                continue
            if not missing:
                print(f"[OK]    {filepath}")
            else:
                print(f"[ISSUES] {filepath}")
                for path, csv_col in missing:
                    print(f"        {path}  ->  '{csv_col}' (not in CSV)")
        print()

    return issues_by_file


def get_meta_cols(df, params):
    """Build a 1-row dataframe of per-participant metadata for the scores output.

    For each entry in `params['metacols']`, if the corresponding column exists
    in the per-trial dataframe `df`, copy the first row's value into the
    returned 1-row dataframe. Keys starting with `_` (e.g. `_comment`
    annotations in a JSON config) are skipped so users can document their
    configs inline.
    """
    metacols_df = pd.DataFrame(index=[0])

    for metacol in params['metacols']:
        if metacol.startswith('_'):
            continue
        if params['metacols'][metacol] and metacol in df.columns:
            metacols_df[metacol] = df[metacol].head(1).values[0]

    return metacols_df.reset_index(drop=True)