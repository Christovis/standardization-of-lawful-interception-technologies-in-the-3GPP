import os
from typing import List, Union
from collections import defaultdict
import pandas as pd
import numpy as np

DOMAIN_DATA_DIR = "/Users/christovis/Documents/InternetGovernance/bigbang/bigbang/datasets/organizations/"
DOMAIN_DATA_FILENAME = "organization_categories.csv"


def load_data():
    """
    Returns a datafarme with email domains labeled by category.

    Categories include: generic, personal, company, academic, sdo

    Returns
    -------
    data: pandas.DataFrame
    """
    domain_data_path = os.path.join(DOMAIN_DATA_DIR, DOMAIN_DATA_FILENAME)
    df = pd.read_csv(
        domain_data_path,
        sep=",",
        header=0,
        index_col=False,
    )
    return df


def remove_leading_and_trailing_whitespaces(df: pd.DataFrame) -> pd.DataFrame:
    for idx, row in df.iterrows():
        for col in row.index:
            if isinstance(row[col], str):
                df.loc[idx, col] = row[col].strip()
    return df


def expand_rows_with_multiple_entries(
    df: pd.DataFrame, column: str='email domain names',
) -> pd.DataFrame:
    # find entries with multiple domain names
    indices = []
    for idx, row in df.iterrows():
        if isinstance(row[column], str):
            if len(row[column].split(',')) > 1:
                indices.append(idx)

    # split entries between single and multiple domain names
    df_multi = df.loc[indices]
    df_single = df.drop(indices)

    # duplicate rows with multiple domain names such that each rows has a single entry
    _df_multi = defaultdict(list)
    for idx, row in df_multi.iterrows():
        _row = row
        entries = row[column].split(',')
        for entry in entries:
            _row[column] = entry.strip()
            for key, value in _row.to_dict().items():
                _df_multi[key].append(value)
    df = pd.concat(
        [df_single, pd.DataFrame.from_dict(_df_multi)],
        ignore_index=True,
    )
    return df


def assign_parent_nationality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change nationality to always be of parent organisation
    e.g. Nokia in USA is still as having a finnish nationality
    """
    # select all rows that have a parent organisation
    _df = df.dropna(subset=['subsidiary of / alias of'])

    for idx, row in _df.iterrows():
        parent = row['subsidiary of / alias of']

        nationality_of_parent = list(df[df['name'] == parent]['nationality'].values)

        if isinstance(nationality_of_parent, list) and (len(nationality_of_parent) == 1):
            nationality_of_parent = list(set(nationality_of_parent))[0]
            df.loc[idx, 'nationality'] = nationality_of_parent
        elif isinstance(nationality_of_parent, list) and (len(nationality_of_parent) == 0):
            # nationality of parent not known
            continue
    return df
