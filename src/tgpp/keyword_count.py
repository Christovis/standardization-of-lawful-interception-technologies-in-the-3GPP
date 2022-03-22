import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils import (
    stem_and_lemmatize,
    text_preprocessing,
    get_bigrams_m1,
    get_bigrams_m3,
)

from bigbang.analysis.listserv import ListservMailList

# Load keywords
file_path_keywords = "../../proj1_3gpp_and_comp/keywords/key_monogram.txt"
keywords = []
with open(file_path_keywords) as file:
    while (line := file.readline().rstrip()):
        keywords.append(line)
keywords = stem_and_lemmatize(keywords)
keywords = list(set(keywords))
keywords.append("privacy")
print(keywords)
# Load mailing-list file paths
mlistdom_path = "../../bigbang-archives/3GPP/"
file_paths = glob.glob(mlistdom_path + "*.mbox")

# Search for keywords
findings = {}
for file_path in file_paths:
    mlist_name = file_path.split("/")[-1].split(".")[0]
    if mlist_name == "3GPP_TSG_SA_WG3_LI":
        mlist = ListservMailList.from_mbox(
            name=mlist_name,
            filepath=file_path,
            include_body=True,
        )
        
        for indx in tqdm(range(len(mlist.df.index)), ascii=True, desc=mlist.name):
            findings[indx] = {}
            findings[indx] .update({
                'date': mlist.df.loc[indx, 'date'],
                'url': mlist.df.loc[indx, 'archived-at'],
            })
            findings[indx].update({
                kw: mlist.df.loc[indx, 'body'].count(kw) for kw in keywords
            })
            #findings[f"{mlist.name}_{indx}"] = {
            #    kw: mlist.df.loc[indx, 'body'].count(kw) for kw in keywords
            #}

findings = pd.DataFrame.from_dict(findings).T
print(findings.columns)
findings = findings.loc[(findings[list(keywords)] != 0).any(axis=1), :]  # remove rows containg only zeros
#findings = findings.loc[:, (findings[list(keywords)] != 0).any(axis=0)]  # remove columns containg only zeros
findings.to_csv("../data/unigram_count_3GPP_TSG_SA_WG3_LI.csv")
