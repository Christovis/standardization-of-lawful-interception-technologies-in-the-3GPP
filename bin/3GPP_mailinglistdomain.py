import os
import glob
import re
import time
import yaml
import json
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


# find available nr. of cpus for parallel computation
ncpus_available = multiprocessing.cpu_count()

parser = argparse.ArgumentParser(
    description="""
    Find target set within search set of documents.
    """,
)
parser.add_argument(
    "--files",
    const="/Users/christovis/Documents/InternetGovernance/bigbang-archive/3GPP/*.mbox",
    default="/Users/christovis/Documents/InternetGovernance/bigbang-archive/3GPP/*.mbox",
    type=str,
    nargs='?',
    help='Define the search set which can be [emails]',
)
parser.add_argument(
    "--ncpus",
    const=5,
    default=5,
    type=str,
    nargs='?',
    help='Number of CPUs to use.',
)
args = parser.parse_args()


def domain_activity(
    file_name: str,
) -> list:
    # load mailinglist
    mlist_name = file_name.split('/')[-1].split('.')[0]
    mlist = ListservMailList.from_mbox(
        name=mlist_name,
        filepath=file_name,
        include_body=False,
    )
    print(f"The mailing list {mlist_name} containes {len(mlist.df.index)} Emails.")
    dic = mlist.get_messagescount(
        header_fields=['from'],
        per_address_field='domain',
        per_year=True,
    )
    dic = dic['from']
    file_path_results = "/Users/christovis/Documents/InternetGovernance/InSight_kickoff/"
    file_handle = open(f"{file_path_results}{mlist_name}.yaml", "w")
    yaml.dump(dic, file_handle)


def main(args):
    file_names_mlist = glob.glob(args.files)
    # run through messages and count keyterms
    time_start = time.time()
    if args.ncpus == 1:
        tset =  [
            domain_activity(file_name)
            for file_name in file_names_mlist
        ]
    else:
        tset = Parallel(n_jobs=args.ncpus)(
            delayed(domain_activity)(file_name)
            for file_name in file_names_mlist
        )


if __name__ == "__main__":
    main(args)
