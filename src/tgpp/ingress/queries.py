from typing import List, Union
import pandas as pd
from sdo_and_cc.ingress import TextFile

from tgpp.nlp.utils import text_preprocessing

def load_queries(file_paths: Union[str, list]) -> List[str]:
    """
    Read quries from .txt file(s).
    """
    if isinstance(file_paths, str):
        queries = TextFile.from_file(file_paths)
    elif isinstance(file_paths, list):
        queries = []
        for file_path in file_paths:
            queries += TextFile.from_file(file_path)
    print(f"There are {len(queries)} queries.")
    return queries


def load_abbreviations(file_paths: Union[str, list]) -> List[str]:
    """
    Read quries from .txt file(s).
    """
    if isinstance(file_paths, str):
        abbreviations = pd.read_csv(file_paths, header=0)
        queries = list(abbreviations.iloc[:, 0].values)
        queries += list(abbreviations.iloc[:, 1].values)
    elif isinstance(file_paths, list):
        queries = []
        for file_path in file_paths:
            abbreviations = pd.read_csv(file_path, header=0)
            queries += list(abbreviations.iloc[:, 0].values)
            queries += list(abbreviations.iloc[:, 1].values)
    print(f"There are {len(queries)} queries.")
    return queries


def remove_text_wo_query(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove texts that contain no queries.

    Parameters
    ----------
    df:
    """
    non_query_columns = [col for col in df.columns if col.startswith('msg-')]
    _df = df.loc[:, ~df.columns.isin(non_query_columns)]
    _df = _df.loc[(_df!=0).any(axis=1), :]  # remove rows with only zeros
    columns = list(_df.columns) + non_query_columns
    indices = list(_df.index)
    df = df.loc[indices, columns].reset_index(drop=True)
    return df


def remove_query_wo_text(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove queries that are not in any text.

        Parameters
        ----------
        df:
        """
        non_query_columns = [col for col in df.columns if col.startswith('msg-')]
        _df = df.loc[:, ~df.columns.isin(non_query_columns)]
        _df = _df.loc[:, (_df!=0).any(axis=0)]  # remove columns with only zeros
        columns = list(_df.columns) + non_query_columns
        indices = list(_df.index)
        df = df.loc[indices, columns].reset_index(drop=True)
        return df
