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
    # load reference set from documents
    # file_paths = glob.glob(CONFIG.folder_reference_set + "*.txt")
    # if file_paths[0].endswith('txt'):
    #     reference_set = TextFile.from_files(file_paths=file_paths)
    # if file_paths[0].endswith('pdf'):
    #     reference_set = PDFFile.from_file(file_path=file_paths)

    # load reference set from preliminary Email selection
    first_target_set = pd.read_hdf(
        CONFIG.folder_target_set + f"{args.search_set}.h5",
        key="df",
        header=0,
        index_col=0,
    )

    # load search set
    search_set = ListservMailList.from_mbox(
        name=args.search_set,
        filepath=f"{CONFIG.folder_search_set}{args.search_set}.mbox",
        include_body=True,
    ).df

    reference_set = search_set[
        search_set['archived-at'].isin(
            list(first_target_set['msg-archived-at'].values)
        )
    ]
    search_set = search_set[
        ~search_set['archived-at'].isin(
            list(first_target_set['msg-archived-at'].values)
        )
    ]
    print("reference_set = ", len(reference_set))
    print("search_set = ", len(search_set))

    # text pre-processing
    text_preprocessing = partial(
        NLPutils.text_preprocessing,
        min_len=2,
        max_len=40,
        keep_nonalphanumerics=['-'],
        remove_numbers=True,
        do_lemmatize=True,
        do_stemming=False,
        return_tokens=False,
    )
    # reference_set = {
    #     name: text_preprocessing(text)
    #     for name, text in reference_set.items()
    # }
    reference_set = {
        row['archived-at']: text_preprocessing(row['body'])
        for idx, row in reference_set.iterrows()
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
    stats_T = stats_T.iloc[:1000]
    stats_ST = stats_ST.iloc[:1000]
    stats_T.to_csv('keyterms_for_T.csv')
    stats_ST.to_csv('keyterms_for_ST.csv')
