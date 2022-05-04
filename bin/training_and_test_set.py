import os, glob
from itertools import compress
from functools import partial
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import pandas as pd

from bigbang.analysis.listserv import ListservMailList

from tgpp.config.config import CONFIG
from tgpp.ingress import TextFile, PDFFile
import tgpp.ingress.queries as Queries
import tgpp.nlp.utils as NLPutils
import tgpp.nlp.query_extractor as QE


parser = argparse.ArgumentParser(
    description="Find target set within search set of documents.",
)
parser.add_argument(
    "--search_set",
    const=CONFIG.search_set,
    default=CONFIG.search_set,
    type=str,
    nargs='?',
    help='Define the search set which can be [emails]',
)
parser.add_argument(
    "--ncpus",
    const=CONFIG.ncpus,
    default=CONFIG.ncpus,
    type=str,
    nargs='?',
    help='Number of CPUs to use.',
)
args = parser.parse_args()


if __name__ == "__main__":
    # file_paths = glob.glob(CONFIG.folder_reference_set + "*.pdf")
    # reference_set = PDFFile.from_file(file_path=file_paths)

    file_paths = glob.glob(CONFIG.folder_reference_set + "*.txt")
    reference_set = TextFile.from_files(file_paths=file_paths)

    search_set = ListservMailList.from_mbox(
        name=args.search_set,
        filepath=f"{CONFIG.folder_search_set}{args.search_set}.mbox",
        include_body=True,
    ).df

    text_preprocessing = partial(
        NLPutils.text_preprocessing,
        min_len=2,
        max_len=40,
        keep_nonalphanumerics=['-'],
        remove_numbers=True,
        return_tokens=False,
    )

    reference_set = {
        name: text_preprocessing(text)
        for name, text in reference_set.items()
    }
    search_set = {
        row['archived-at']: text_preprocessing(row['body'])
        for idx, row in search_set.iterrows()
    }

    stats_T, stats_ST = QE.using_klr(
        reference_set=reference_set,
        search_set=search_set,
        classifiers="nbayes",
    )
    print(stats_T)
