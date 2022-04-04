import re
import random
from tqdm import tqdm
from typing import List, Dict, Union, Tuple
from collections import namedtuple
import numpy as np
from scipy import sparse

import nltk
#from nltk import word_tokenize
#from nltk.collocations import BigramCollocationFinder
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sdo_and_cc.ingress import TextFile
from sdo_and_cc.config.config import CONFIG

#nltk.download("omw-1.4")
#nltk.download("wordnet")
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
STOPWORDS = TextFile.from_file(CONFIG.file_stopwords)


def lemmatize(tokens: List[str]) -> List[str]:
    return [lemmatizer.lemmatize(token, pos="v") for token in tokens]


def stemming(tokens: List[str]) -> List[str]:
    return [stemmer.stem(token) for token in tokens]


def filter_words(
    tokens: List[str], min_len: int=2, max_len: int=15,
) -> List[str]:
    return [
        token
        for token in tokens
        if (min_len <= len(token) <= max_len) and (token not in STOPWORDS)
    ]


def tokenize_text(text: str) -> List[str]:
    return text.split()


def text_preprocessing(
    text: Union[List[str], str],
    min_len: int = 2,
    max_len: int = 15,
    remove_punctuations: bool=True,
    remove_numbers: bool=True,
    return_tokens: bool=True,
) -> Union[List[List[str]], List[str]]:
    """
    A text is the ascii content of a document, which is being pre-processed using:
        - String to lower case
        - Tokenize text
        - Remove string shorter than three and longer than 15 characters
        - Remove stop words and words shorter than two characters
        - Lemmatization and Stemming

    Parameters
    ----------
    text:
    min_len:
    max_len:
    return_tokens:
    """
    if isinstance(text, str):
        # everything lower case
        text = text.lower()
        # remove unicode characters
        text = text.encode("ascii", "ignore").decode()
        if remove_numbers:
            text = re.sub("[0-9]", " ", text)
        if remove_punctuations:
            text = re.sub(r'[!"#$%&()*+,\-./:;<=>?@[\\\]^_`{|}~\']', " ", text)
        tokens = tokenize_text(text)
        tokens = filter_words(tokens, min_len, max_len)
        tokens = lemmatize(tokens)
        tokens = stemming(tokens)
    if isinstance(text, list):
        tokens = [
            text_preprocessing(
                t,
                min_len,
                max_len,
                remove_punctuations,
                remove_numbers,
                return_tokens=False,
            )
            for t in text
        ]
    if return_tokens:
        return tokens
    else:
        return (" ").join(tokens)


def corpus_preprocessing(
    corpus: Dict[str, str],
    **args,
) -> Dict[str, Union[List[str], str]]:
    """
    A corpus is a dictionary of the form {'name': text, ...} that stands for
    a archive of documents. A single document a dictionary of the form
    {'name': text} with only one key.
    """
    return {
        doc_name: text_preprocessing(doc_text, **args)
        for doc_name, doc_text in tqdm(
            corpus.items(), ascii=True, desc="Preprocessing Text"
        )
    }


def get_diff_of_sets(
    set_a: List[str],
    set_b: str,
    return_indices: bool = False,
) -> List[Union[str, int]]:
    if return_indices is False:
        return [ngram for ngram in set_a if ngram not in set_b]
    else:
        return [indx for indx, ngram in enumerate(set_a) if ngram not in set_b]


def create_training_and_test_sets(
    reference_set: Dict[str, str],
    search_set: Dict[str, str],
    ref_frac: float = 1.0,
    search_frac: float = 0.33,
    rnd_seed: int = 12345,
) -> Tuple:
    """
    Note: The documents in the reference_set and search_set are not tokenized.

    Parameters
    ----------
    reference_set: A set of pre-processed textual documents, all of which are
        examples of a single chosen concept of interest
        (e.g., topic, sentiment, idea, person, organization, event).
    search_set: A set of pre-processed documents selected because it likely has
        additional documents of interest, as well as many others not of interest.
    ref_frac, search_frac : Fraction of the reference_set and search_set that is
        used for their training sets.
    """
    cl_input = get_dt_matrix(dict(reference_set, **search_set))

    # fix random seed for reproducability
    random.seed(rnd_seed)
    n_doc_in_ref = len(reference_set)
    n_doc_in_sea = len(search_set)

    # split reference set in training and test set
    ref_train_keys = random.sample(
        list(list(reference_set.keys())),
        int(n_doc_in_ref * ref_frac),
    )
    ref_test_keys = random.sample(
        list(list(reference_set.keys())),
        int(n_doc_in_ref * ref_frac),
    )
    # for key in ref_train_keys:
    #    del reference_set[key]
    # ref_test_keys = list(reference_set.keys())
    print(
        f"{len(ref_train_keys)} ref-docs in train and"
        + "{len(ref_test_keys)} ref-docs in test."
    )

    # split search set in training and test set
    search_train_keys = random.sample(
        list(list(search_set.keys())),
        int(n_doc_in_sea * search_frac),
    )
    for key in search_train_keys:
        del search_set[key]
    search_test_keys = list(search_set.keys())
    print(
        f"{len(search_train_keys)} sea-docs in train and"
        + "{len(search_test_keys)} sea-docs in test."
    )

    # test set
    test_keys = ref_test_keys + search_test_keys
    # test_keys = list(search_set.keys())
    indices = np.array(
        [index for index, did in enumerate(cl_input.docs) if did in test_keys]
    )
    x_test = cl_input.matrix[indices, :]
    # training set
    train_keys = ref_train_keys + search_train_keys
    indices = np.array(
        [index for index, did in enumerate(cl_input.docs) if did in train_keys]
    )
    x_train = cl_input.matrix[indices, :]
    # value to be predicted
    y_train = []
    for index in indices:
        if cl_input.docs[index] in ref_train_keys:
            y_train.append(0)
        elif cl_input.docs[index] in search_train_keys:
            y_train.append(1)
    return y_train, x_train, x_test, test_keys


def get_dt_matrix(corpus: Dict[str, str], **kwargs) -> Tuple:
    """
    Parameters
    ----------

    Returns
    -------
    """
    input_object = namedtuple("classifier_input", "matrix terms docs")
    vectorizer = CountVectorizer(
        min_df=0,  # keep rare terms, as concept of interest is niche
        max_df=0.9,  # as concept of interest is niche, ignore frequent terms
        tokenizer=tokenize_text,
        decode_error="ignore",
        ngram_range=(2, 2),  # only bigrams
        **kwargs,
    )
    dt_matrix = vectorizer.fit_transform(list(corpus.values()))  # .toarray()
    terms = vectorizer.get_feature_names()
    return input_object(
        matrix=dt_matrix, terms=terms, docs=list(corpus.keys())
    )


def sample_dt_matrix(dt_matrix: namedtuple, docs: list) -> namedtuple:
    indices = np.array(
        [index for index, dn in enumerate(dt_matrix.docs) if dn in docs],
        dtype=np.int8,
    )
    print(indices)
    input_object = namedtuple("classifier_input", "matrix terms docs")
    return input_object(
        matrix=sparse.csr_matrix(dt_matrix.matrix.A[indices, :]),
        terms=dt_matrix.terms,
        docs=docs,
    )
