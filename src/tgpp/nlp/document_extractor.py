from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

import tgpp.nlp.concept_extractor as CE


def using_cosine_similarity(
    text: List[List[str]],
    query: List[str],
    word_limit: Optional[int]=None,
    ngram: int=2,
) -> Dict:
    """
    Use cosine similarity to assess relevance of documents w.r.t. search query.

    Based on:
        Introduction to Information Retrieval (2008), Eq. 6.12
    """
    return cosine_similarity(query, text)


def using_tfidf(
    text: List[List[str]],
    query: List[str],
    word_limit: Optional[int]=None,
    ngram: int=2,
) -> Dict:
    """
    Use Term Frequency – Inverse Document Frequency to assess relevance of
    documents w.r.t. search query.

    Based on:
        Introduction to Information Retrieval (2008), Eq. 6.9, Fig. 6.14

    Note:
        N : number of documents
        q : query
        d : document
        t : term

    Parameters
    ----------
    text: Pre-processed text from which concepts have to be extracted.
    """
    vectorizer = TfidfVectorizer(
        min_df=3, max_df=0.95,
        max_features=n_features,
        token_pattern="\w+\$*\d*",
    )
    tfidf_matrix = vectorizer.fit_transform(text).toarray()
    score = _overlap_score_measure(tfidf_matrix)
    score = score / len(document)
    return score


def _overlap_score_measure(
    documents: List[str],
) -> np.array:
    """
    Add up all tf-idf weight of each term in document, d.
    """
    tfidf = using_tfidf(documents)
    score = tfidf.sum(axis=0)
    return score
