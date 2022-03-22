import os
import re
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
from itertools import compress
from collections import defaultdict
import pandas as pd

from bigbang.analysis.listserv import ListservMailList

from tgpp.ingress import TextFile
from tgpp.nlp.utils import text_preprocessing

folder_project = str(Path(os.path.abspath(__file__)).parent.parent)
folder_data = folder_project + "/data"
folder_keys = folder_project + "/keywords"
folder_bin = folder_project + "/bin"
#folder_emails = "/home/christovis/InternetGov/bigbang-archives/3GPP"
folder_emails = "/home/christovis/InternetGov/bigbang/archives/3GPP"

logging.basicConfig(
    filename=folder_bin + "/find_target_documents.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="""
    Find target set within search set of documents.
    """,
)
parser.add_argument(
    "--query_file",
    const=folder_project + '/keywords/bigrams_unsupervised_verified.csv',
    default=folder_project + '/keywords/bigrams_unsupervised_verified.csv',
    type=str,
    nargs='?',
    help='File-path to file containing search queries.',
)
parser.add_argument(
    "--search_set",
    const='email',
    default='email',
    type=str,
    nargs='?',
    help='Define the search set which can be [email, rfcfinal, rfcrecent]',
)
args = parser.parse_args()


def queries_from_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        header=0,
        index_col=0,
    )
    return list(df['ngram'].values)

def main(args):
    queries = queries_from_csv(args.query_file)
    queries += text_preprocessing(
        TextFile.from_file(f"{folder_keys}/key_bigrams.txt"),
        min_len=2,
        return_tokens=True,
    )
    queries += text_preprocessing(
        TextFile.from_file(f"{folder_keys}/key_unigrams.txt"),
        min_len=2,
        return_tokens=True,
    )
    queries = list(set(queries))
    # add acronyms
    queries += [
        " eid ",
        " nef ",
        " nesas ",
        " rfid ",
        " e2e ",
    ]
    queries = [q for q in queries if len(q) >= 3]

    mlist_name = "3GPP_TSG_SA_WG3_LI"
    mlist = ListservMailList.from_mbox(
        name=mlist_name,
        filepath=f"{folder_emails}/{mlist_name}.mbox",
    )
    print(f"The mailing list containes {len(mlist.df.index)} Emails.")

    col_attachment = [
        col for col in mlist.df.columns if col.startswith('attachment-')
    ]
    findings = defaultdict(list)
    for msg_idx, msg in mlist.df.iterrows():
        text = '' #msg.body
        if len(msg[col_attachment].dropna()) > 0:
            attachment = ('Attachment:\n').join(list(msg[col_attachment].dropna().values))
            text = text + '\n' + attachment
        text_pp = text_preprocessing(text, min_len=2, return_tokens=False)
        findings['message-attachment'].append(text)
        findings['message-id'].append(msg['message-id'])
        findings['message-from'].append(msg['from'])
        findings['message-date'].append(msg['date'])
        findings['archived-at'].append(msg['archived-at'])
        for query in queries:
            if query == "intercept":
                _text = text_pp.replace('law intercept group', '')
                findings[query].append(_text.count(query))
            else:
                findings[query].append(text_pp.count(query))

    query_name = args.query_file.split('/')[-1].split('.')[0]
    file_name = f"{folder_data}/{query_name}_in_maillist_attachments.csv"
    df = pd.DataFrame.from_dict(findings)

    print(df.columns)
    # only keep non-zero rows and columns
    non_query_columns = [
        'message-attachment',
        'message-id',
        'message-date',
        'message-from',
        'archived-at',
    ]
    _df = df.loc[:, ~df.columns.isin(non_query_columns)]
    _df = _df.loc[(_df!=0).any(axis=1), :]  # remove rows with only zeros
    _df = _df.loc[:, (_df!=0).any(axis=0)]  # remove columns with only zeros
    columns = list(_df.columns) + non_query_columns
    indices = list(_df.index)
    df = df.loc[indices, columns]
    print(df.columns)

    # remove row with only one unigram count
    columns_unigrams = [col for col in df.columns if len(col.split(' ')) == 1]
    _df = df[columns_unigrams]
    _df = _df.loc[:, ~_df.columns.isin(non_query_columns)]
    _df = _df.loc[_df.sum(axis=1) > 1, :]
    indices = list(_df.index)
    df = df.loc[indices, columns]
    print(df.columns)

    df.to_csv(file_name)


if __name__ == "__main__":
    main(args)
