'''
Finger Oscillation (FO) scoring.

The Finger Oscillation task has 2 blocks (dominant hand, then non-dominant hand).
In each block, participants tap as fast as they can — either pressing a key or
clicking a button on a touchscreen — for a fixed number of taps. One CSV row
per tap. This module reads PsychoPy CSV output and emits a per-tap trials
dataframe plus per-block mean/SD of response time.

Two CSV shapes are supported through the per-block config:
  - "wide" (test1/2/3): each block has its own column names (e.g.
    `dominant_key_resp.rt` vs `nondominant_key_resp.rt`). Block membership is
    inferred from which trial column is non-null — same pattern as pgng.
  - "stacked" (test4): both blocks share the same column names, and a
    `block_index` (0 or 1) plus a `block_marker_col` per block tells the scorer
    how many block-end markers must precede a row for it to belong to that block.

Note on hand-dominance labelling: instructions to participants may have been
inconsistent across sessions. This module trusts the labels you set in your
config (e.g. `metavars.hand = 'dominant'`) — verification, if you want it, is
something you do downstream by comparing per-block RTs.
'''

import logging
import os
import pandas as pd
import numpy as np
from mend2np.utils import (
    setup_logger,
    get_meta_cols,
    handle_multiple_responses,
    validate_params,
    run_task,
    copy_configured_columns,
)

# Module-level logger reference so helper functions can `logger.warning(...)`
# without a `global` declaration. setup_logger configures the package 'mend2np' logger.
logger = logging.getLogger(__name__)

REQUIRED_PARAMS = {
    'metacols': dict,
    'blocks': dict,
}


def fingosc(params:dict, out:str=os.getcwd(), write:bool=True, filelist:str|list='',
            formatted:bool=False, log=20, logfile:bool=False, trial_filter:str='') -> tuple:
    """Score one or more Finger Oscillation data files.

    :param params: configuration dict. Must include `metacols` (CSV column ↔ metadata
        name mapping) and `blocks` (a per-block dict of `cols` and optional `metavars`).
        See `tests/fingosc_example.json` for the typical shape.
    :param out: directory to write output CSVs (created if missing).
    :param write: if True, write combined trial- and score-level CSVs.
    :param filelist: list of CSV paths, path to a text file with one CSV path per
        line, or empty for GUI picker.
    :param formatted: True if the input is already tidy with the library's standard
        column names; default False (raw PsychoPy output).
    :param log: log level.
    :param logfile: if True, write a timestamped ``log_<ts>.log`` to ``out`` (default False).
    :param trial_filter: optional pandas query string applied before scoring.
    :returns: `(combined_scores, combined_trials)`.
    """
    setup_logger(out=out, level=log, logfile=logfile).info('start')
    validate_params(params, REQUIRED_PARAMS)

    def process_one(filepath, params, logger):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        if not formatted:
            df = format_df(df, params)
        df.insert(1, 'filename', filename)
        scores_row = pd.concat([get_meta_cols(df, params), score_df(df, trial_filter)], axis=1)
        scores_row.insert(1, 'filename', filename)
        return df, scores_row

    return run_task(
        params=params, filelist=filelist, out=out, write=write,
        process_file_fn=process_one,
    )


def format_df(df:pd.DataFrame, params:dict) -> pd.DataFrame:
    """Reshape a raw FO CSV into the library's standard column layout.

    Walks `params['blocks']`. For each block, builds a row mask, copies metacols
    and trial-level cols (via `copy_configured_columns`, which warns when a
    configured CSV column is missing), broadcasts any static `metavars`, and
    stamps the block key onto every row. Returns the concatenated per-block
    dataframes.
    """
    fmtdf = pd.DataFrame()

    for block_key, block_cfg in params['blocks'].items():
        if isinstance(block_key, str) and block_key.startswith('_'):
            continue

        block_cols = block_cfg.get('cols', {})
        block_metavars = block_cfg.get('metavars', {})

        # Build the row mask for this block.
        mask = _block_mask(df, block_cols, block_metavars, block_key)
        if mask is None:
            # _block_mask emits a warning already; just skip this block.
            continue

        tmpdf = pd.DataFrame()
        copy_configured_columns(tmpdf, df, block_cols,
                                f'blocks.{block_key}.cols', mask=mask, logger=logger)

        # Some FO variants (touch) store response/rt cells as list-strings —
        # parse them to lists. Other variants pass through unchanged.
        for resp_col in ['response', 'rt']:
            if resp_col in tmpdf.columns:
                tmpdf[resp_col] = tmpdf[resp_col].apply(
                    lambda x: handle_multiple_responses(x, slice_index=slice(None))
                )

        # Coerce rt to numeric where possible (touch CSVs sometimes encode it as a
        # string list, which `handle_multiple_responses` flattens; non-list scalar
        # strings get cleaned and converted here).
        if 'rt' in tmpdf.columns:
            tmpdf['rt'] = pd.to_numeric(tmpdf['rt'], errors='coerce')

        # Static metavars (e.g. `hand: 'dominant'`) broadcast onto every row.
        # `block_index` and `block_marker_col` are control flags used by
        # `_block_mask`; don't broadcast them into the output trials dataframe.
        for metavar, value in block_metavars.items():
            if metavar in ('block_index', 'block_marker_col'):
                continue
            if metavar.startswith('_'):
                continue
            tmpdf[metavar] = value

        # Per-participant metadata: broadcast onto every row. Done AFTER tmpdf has
        # rows so the assignment broadcasts instead of producing an empty column
        # (the same trap that bit pgng's metacols; see pgng.format_df comment).
        for metacol in params['metacols']:
            if isinstance(metacol, str) and metacol.startswith('_'):
                continue
            csv_col = params['metacols'][metacol]
            if not csv_col:
                continue
            if csv_col in df.columns:
                idx = df[csv_col].first_valid_index()
                tmpdf[metacol] = df.at[idx, csv_col] if idx is not None else None
            else:
                logger.warning(
                    f"metacols.{metacol}: configured CSV column '{csv_col}' is not "
                    f"in this file's columns — '{metacol}' will be missing for "
                    f"block {block_key}."
                )

        # Reorder so metacols come before per-block cols (matches the layout of
        # the other tasks' output).
        metacol_names = [m for m in params['metacols'] if m in tmpdf.columns
                         and not (isinstance(m, str) and m.startswith('_'))]
        other_cols = [c for c in tmpdf.columns if c not in metacol_names]
        tmpdf = tmpdf[metacol_names + other_cols]

        # Stamp the block key onto every row (used by `score_df` for grouping).
        tmpdf['block'] = block_key

        fmtdf = pd.concat([fmtdf, tmpdf], ignore_index=True)

    return fmtdf


def _block_mask(df:pd.DataFrame, block_cols:dict, block_metavars:dict, block_key) -> pd.Series:
    """Build a boolean row mask selecting this block's trials from `df`.

    Two strategies:

      - **Wide** (the default): mask = `df[trial_col].notna()`, exactly the pgng
        pattern. Used when each block has its own `cols.trial` value.
      - **Stacked**: when `block_metavars` carries `block_index` (integer) and
        `block_marker_col` (a CSV column whose non-null cells mark each block's
        END), trials are assigned to a block by counting how many marker rows
        precede them. Used for variants like test4 where both blocks share the
        same trial column.

    Returns None and logs a warning when no usable mask can be built.
    """
    trial_col = block_cols.get('trial')
    block_index = block_metavars.get('block_index')
    marker_col = block_metavars.get('block_marker_col')

    # Stacked mode: requires both block_index and block_marker_col to be set.
    if block_index is not None and marker_col is not None:
        if marker_col not in df.columns:
            logger.warning(
                f"blocks.{block_key}: configured block_marker_col '{marker_col}' "
                f"is not in this file's columns; cannot split blocks. Skipping."
            )
            return None
        # `cumsum().shift(1)`: count markers *before* each row (so the marker row
        # itself counts for the block it terminates, not for the next block).
        markers = df[marker_col].notna().astype(int)
        cum_markers = markers.cumsum().shift(1).fillna(0).astype(int)
        mask = (cum_markers == int(block_index))
        # Also require trial_col non-null if it's configured — filters out
        # non-trial rows (e.g. headers, intros, the block-end marker row itself).
        if trial_col and trial_col in df.columns:
            mask &= df[trial_col].notna()
        return mask

    # Wide mode: rely on the per-block trial column being non-null.
    if trial_col and trial_col in df.columns:
        return df[trial_col].notna()

    logger.warning(
        f"blocks.{block_key}: neither a non-null trial column nor "
        f"(block_index + block_marker_col) configured; cannot build a row mask. "
        f"Skipping this block."
    )
    return None


def score_df(df:pd.DataFrame, trial_filter:str) -> pd.DataFrame:
    """Build a 1-row dataframe of per-file FO scores.

    For each block (grouped by the `block` column written in `format_df`),
    emits `<block>_mean_rt` and `<block>_sd_rt` columns.

    :param df: per-trial dataframe.
    :param trial_filter: optional pandas query string applied before scoring.
    """
    if trial_filter:
        df = df.query(trial_filter)

    score_dict = {}
    if 'block' not in df.columns or 'rt' not in df.columns:
        return pd.DataFrame(score_dict, index=[0])

    for block_key, blk in df.groupby('block'):
        score_dict[f'{block_key}_n_trials'] = len(blk)
        score_dict[f'{block_key}_mean_rt'] = blk['rt'].mean()
        score_dict[f'{block_key}_sd_rt'] = blk['rt'].std()

    return pd.DataFrame(score_dict, index=[0])
