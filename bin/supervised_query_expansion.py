import os
import re
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

import nltk

from bigbang.analysis.listserv import ListservMailList

from tgpp.config.config import CONFIG
from tgpp.ingress import PolyFiles, RFCFinal, EmailList
from tgpp.nlp.utils import (
    text_preprocessing,
    corpus_preprocessing,
)
import tgpp.nlp.query_extractor as QE


# Load search set, S, for the KLR method
sset_selection = list(pd.read_csv(
    f"{CONFIG.folder_target_set}bigrams_unsupervised_verified_in_maillist.csv",
    header=0,
    index_col=0,
)['email_id'].values)
mlist_name = "3GPP_TSG_SA_WG3_LI"
mlist = ListservMailList.from_mbox(
    name=args.search_set,
    filepath=f"{CONFIG.folder_search_set}{args.search_set}.mbox",
)
sset = mlist.df[mlist.df['message-id'].isin(sset_selection)]
sset = sset[['message-id', 'body']].set_index('message-id').T.to_dict('list')
sset = corpus_preprocessing(sset, return_tokens=False)

result_T, result_ST = QE.using_klr(
    refset, seaset,
    classifiers=["nbayes", "logit"],
    ref_frac=1.0,
    search_frac=.33,
)

#result_T.to_csv(CONFIG.folder_keyterms + "bigrams_supervised.csv")
