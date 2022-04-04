import os
import re
import time
from functools import partial
from typing import List, Dict
from pathlib import Path
import argparse
from tqdm import tqdm
from itertools import compress
from collections import defaultdict
import pandas as pd
import numpy as np
from collections import namedtuple
import multiprocessing
from joblib import Parallel, delayed

from bigbang.analysis.listserv import ListservMailList

from tgpp.config.config import CONFIG
from tgpp.ingress import TextFile
import tgpp.ingress.queries as Queries
from tgpp.nlp.utils import text_preprocessing

# find available nr. of cpus for parallel computation
ncpus_available = multiprocessing.cpu_count()

parser = argparse.ArgumentParser(
    description="""
    Find target set within search set of documents.
    """,
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

text_preprocessing = partial(
    text_preprocessing,
    min_len=1,
    max_len=30,
    remove_punctuations=True,
    remove_numbers=True,
)


def search_keyterms(
    msg: pd.Series,
    queries: List[str],
    header_fields: List[str],
    attachment_fields: List[str],
) -> list:
    tset_msg = {}
    # get texts of message body and attachment
    body = msg['body']
    attachment = (' ').join(list(msg[attachment_fields].dropna().values))
    # preprocess texts
    preproc_body = text_preprocessing(body, return_tokens=False)
    preproc_attachment = text_preprocessing(attachment, return_tokens=False)
    # add Email header fields
    for header_field in header_fields:
        try:
            tset_msg[f'msg-{header_field}'] = msg[header_field]
        except Exception:
            tset_msg[f'msg-{header_field}'] = None
    # add query counts
    for query in queries:
        tset_msg[f'body-{query}'] = preproc_body.count(query)
        tset_msg[f'attachment-{query}'] = preproc_attachment.count(query)
    # add extra fields
    #tset_msg['msg-body'] = body
    tset_msg['msg-body_token_count'] = len(preproc_body)
    #tset_msg['msg-attachment'] = attachment
    tset_msg['msg-attachment_token_count'] = len(preproc_attachment)
    return list(tset_msg.values())



def main(args):
    # load keyterms/queries
    queries = Queries.load_abbreviations(CONFIG.file_queries)
    queries = text_preprocessing(queries, min_len=1, return_tokens=True)
    # remove dublicates
    queries = list(np.unique(queries))
    # remove empty strings
    queries.remove('')
    queries = [' '+query+' ' for query in queries]

    # load mailinglist
    mlist = ListservMailList.from_mbox(
        name=args.search_set,
        filepath=f"{CONFIG.folder_search_set}{args.search_set}.mbox",
        include_body=True,
    )
    print(f"The mailing list containes {len(mlist.df.index)} Emails.")

    attachment_fields = [
        col for col in mlist.df.columns if col.startswith('attachment-')
    ]

    # run through messages and count keyterms
    time_start = time.time()
    if args.ncpus == 1:
        tset =  [
            search_keyterms(msg, queries, CONFIG.header_fields, attachment_fields)
            for msg_idx, msg in mlist.df.iterrows()
        ]
    else:
        tset = Parallel(n_jobs=args.ncpus)(
            delayed(search_keyterms)(msg, queries, CONFIG.header_fields, attachment_fields)
            for msg_idx, msg in mlist.df.iterrows()
        )
    print(time.time() - time_start)

    # Target-set Email attributes
    attributes = {f'msg-{header_field}': str for header_field in CONFIG.header_fields}
    for query in queries:
        attributes[f'body-{query}'] = int
        attributes[f'attachment-{query}'] = int
    attributes['msg-body_token_count'] = int
    attributes['msg-attachment_token_count'] = int
    tset = np.asarray(tset).T
    tset_msg = {}
    for idx, (attribute, datatype) in enumerate(attributes.items()):
        tset_msg[attribute] = tset[idx].astype(datatype)

    df = pd.DataFrame.from_dict(tset_msg)
    df = Queries.remove_text_wo_query(df)
    df = Queries.remove_query_wo_text(df)
    print(len(df.index), len(df.columns))

    #TODO: Need to specify escapechar as white-space etc can be contianed in text body
    #df.to_csv(_file_path, escapechar=), hdf5 might therefore be better in that case
    _file_path = CONFIG.folder_target_set + f"{args.search_set}.h5"
    df.to_hdf(_file_path, key='df', mode='w')


if __name__ == "__main__":
    main(args)
