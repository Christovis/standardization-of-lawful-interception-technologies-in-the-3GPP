import os
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

import nltk

from tgpp.ingress import PolyFiles, TextFile
from tgpp.nlp.utils import text_preprocessing
import tgpp.nlp.query_extractor as QE

folder_project = str(Path(os.path.abspath(__file__)).parent.parent)
folder_data = folder_project + "/data"
folder_keys = folder_project + "/keywords"
folder_bin = folder_project + "/bin"

# Find frequent bigrams from reference set, R
reference_generator = PolyFiles(
    folder_path=folder_data + "/reference_set",
    file_dsc="*.pdf",
)
reference_tokens = []
for doc in reference_generator:
    reference_tokens += text_preprocessing(
        list(doc.values())[0], return_tokens=True,
    )
bigrams = QE.using_frequency(reference_tokens, freq_limit=10, ngram=2)

# Remove bigrams that appear in unrelated text, S\T
urls = TextFile.from_file(f"{folder_data}/url_topic_outside_focus.txt")
doc_generator = PolyFiles(url=urls)
indices = []
for doc in doc_generator:
    text = text_preprocessing(list(doc.values())[0], return_tokens=False)
    indices += [i for i, bg in enumerate(bigrams['ngram']) if bg not in text]
bigrams['ngram'] = [bigrams['ngram'][i] for i in list(set(indices))]
bigrams['freq'] = [bigrams['freq'][i] for i in list(set(indices))]

# Save bigrams
#df = pd.DataFrame.from_dict(bigrams)
#df = df.sort_values(by=['freq'], ascending=False)
#df.to_csv(folder_keys + "/bigrams_unsupervised.csv")
