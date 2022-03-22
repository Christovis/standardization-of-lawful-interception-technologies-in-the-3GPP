import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import sparse
from math import lgamma
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

from sdo_and_cc.nlp.classifiers import Classifiers as CL
from sdo_and_cc.nlp.utils import (
    get_dt_matrix,
    sample_dt_matrix,
    text_preprocessing,
    create_training_and_test_sets,
)

logger = logging.getLogger(__name__)


def using_frequency(
    text: List[str],
    freq_limit: Optional[int]=None,
    word_limit: Optional[int]=None,
    ngram: int=2,
) -> Dict:
    """
    Find concepts based on the frequency with which they appear in the corpus.
    This expansion works without providing an initial information retrieval query.

    Parameters
    ----------
    text: Pre-processed tokenised text from which concepts have to be extracted.
    freq_limit: If not None, then the concepts with a frequency higher than
        `freq_limit` are returned.
    word_limit: If not None, then the top `word_limit` concepts are returned.
    ngram: The length of the continues sequence of words:
        1 = unigram, 2 = bigram, ...

    Returns
    -------
    """
    ngs = nltk.ngrams(text, ngram)
    ngs = dict(nltk.FreqDist(ngs))
    # order with increasing frequency
    ngs = dict(sorted(ngs.items(), key=lambda item: item[1]))
    if freq_limit:
        ngs = {k: v for k, v in ngs.items() if v >= freq_limit}
    if word_limit:
        ngs_nr = len(ngs)
        ngs = {k: v
            for indx, (k, v) in enumerate(ngs.items())
            if indx >= (ngs_nr - word_limit)
        }
    dic = {'ngram': [], 'freq': []}
    for k, v in ngs.items():
        dic['ngram'].append((' ').join(list(k)))
        dic['freq'].append(v)
    return dic


def using_tfidf(
    text: List[List[str]],
    word_limit: Optional[int]=None,
    ngram: int=2,
) -> Dict:
    """
    Term Frequency – Inverse Document Frequency
    
    Parameters
    ----------
    text: Pre-processed tokenised text from which concepts have to be extracted.
    """
    vectorizer = TfidfVectorizer(
        min_df=3, max_df=0.95,
        max_features=n_features,
        token_pattern="\w+\$*\d*",
    )
    tfidf_matrix = vectorizer.fit_transform(text).toarray()
    return tf * idf


def using_kl_divergence(
    text: List[List[str]],
    query: List[str],
    word_limit: Optional[int]=None,
    alpha: float=1e-6,
    ngram: int=2,
    n_features: int=5000,
) -> Dict:
    """
    A concept expansion that is normally used in topic modelling and is not
    adjusted to a specific corpus.

    Note:
        This expansion required an initial information retrieval query.
    
    Based on:
        Claudio Carpineto et al.  
        "An information theoretic approach to automatic query expansion"
        (2001)

    Parameters
    ----------
    text: Pre-processed text from which concepts have to be extracted.
    query: Pre-processed queries with which relevant texts are identified.
    alpha: Hyperparameter with value = [0, 1] to compute score.
    n_features: Limit to the use of top `n_features` vocabulary

    Note:
    The following notation for variables is used:
        n_ : number of ...
        p_ : probability of ...
        _t_ : term/token/word
        _cor : corpus
        _doc : document
    """
    assert 0 < alpha <= 1, "Alpha has to be within 0 and 1."
    # Compute the document-term matrix
    dt_matrix = get_dt_matrix(text)
    p_t_cor = _term_frequency(dt_matrix, mode='corpus')
    p_t_doc = _term_frequency(dt_matrix, mode='document')
    score = _weighted_zone_scoring(term_freq_corpus, term_freq_doc, alpha)

    prob_t_in_subcorpus = _weighted_zone_scoring(term_freq_corpus, _, alpha)
    score = p_t_subcorpus * np.log2(p_t_subcorpus/p_t_cor)
    terms = terms[np.argsort(score)[:-word_limit - 1:-1]]
    dic = {'ngram': list(terms), 'score': list(score)}
    return dic


def using_klr(
    reference_set: Dict[str, str],
    search_set: Dict[str, str],
    #word_limit: Optional[int]=None,
    classifiers: List[str]=["nbayes", "logit"],
    ref_frac: float=1.0,
    search_frac: float=.33,
    **args,
) -> Dict:
    """
    A concept expansion tailored to a specific corpus that is being analysed.

    Note:
        - This expansion required an reference set and a search set.
        - Since the reference set (R) is typically much smaller than the search set
          (S) and our test set for our classifiers is all of S, we often use the
          entire R set and a sample of S as our training set.

    Based on:
        G King, P L Thresher, M E Roberts
        "Computer-Assisted Keyword and Document Set Discovery from Unstructured Text"
        (2017)

    Parameters
    ----------
    reference_set: A reference set is a set of pre-processed textual documents,
        all of which are examples of a single chosen concept of interest
        (e.g., topic, sentiment, idea, person, organization, event).
    search_set: A search set is a set of pre-processed documents selected because
        it likely has additional documents of interest, as well as many others
        not of interest.
    classifiers:

    Returns
    -------
    """
    
    def log_likelihood_fct(term):
        # number of docs in T containing word
        n1 = dc_T[term]
        # number of docs in S\T containing word
        n0 = dc_S[term] - n1
        # number of docs in S\T not containing word
        n1_n = n_T - n1
        # number of docs in S\T not containing word
        n0_n = n_ST - n0
        # log-likelihood
        ll = (
            lgamma(n1+alpha_T) + lgamma(n0+alpha_ST) - lgamma(n1+alpha_T+n0+alpha_ST)
        ) + (
            lgamma(n1_n+alpha_T) + lgamma(n0_n+alpha_ST) - lgamma(n1_n+alpha_T+n0_n+alpha_ST)
        )
        return ll
    
    y_train, x_train, x_test, key_test = create_training_and_test_sets(
        reference_set,
        search_set,
        ref_frac,
        search_frac,
        **args,
    )
    logger.info("Created train and test set")
    docnames_T, docnames_ST = CL(
        y_train, x_train, x_test, key_test, classifiers,
    )

    if len(docnames_T) == 0:
        return None, None
    else:
        dt_matrix_S = get_dt_matrix(search_set)
        dt_matrix_T = sample_dt_matrix(dt_matrix_S, docnames_T)

        # Counts of documents within the total corpus set that contain word
        dc_S = get_doc_counts_for_term(dt_matrix_S.matrix, dt_matrix_S.terms)
        # Counts of documents within the target set that contain word
        dc_T = get_doc_counts_for_term(dt_matrix_T.matrix, dt_matrix_T.terms)
        # number of target documents in search set, T
        n_T = len(docnames_T)
        if n_T == 0:
            n_T = 1
        # number of non-target documents in search set, S\T
        n_ST = len(docnames_ST)

        # with aplha_T = alpha_ST = 1. in our implementation
        alpha_T = 1
        alpha_ST = 1
        ranked_by = 'll'
        stats = defaultdict(list)
        for term in dt_matrix_S.terms:
            ll = log_likelihood_fct(term)
            # likelihood term rightly categorizes document into target set, T
            p_T = float(dc_T[term]) / n_T
            # likelihood term rightly categorizes document into non-target set, S\T
            p_ST = float(dc_S[term] - dc_T[term]) / n_ST
            
            stats['n_T'].append(p_T * n_T)
            stats['n_ST'].append(p_ST * n_ST)
            stats['p_T'].append(p_T)
            stats['p_ST'].append(p_ST)
            stats['ll'].append(ll)
            stats['term'].append(term)
            if p_ST > p_T:
                stats['category'].append('S\T')
            else:
                stats['category'].append('T')

        # rank keywords from highest to lowest likelihood for T
        stats = pd.DataFrame(stats)
        stats = stats.set_index('term')
        stats_T = stats[stats['category'] == 'T']
        stats_T.sort('ll', ascending=False, inplace=True)
        stats_ST = stats[stats['category'] == 'S\T']
        stats_ST.sort('ll', ascending=False, inplace=True)

        del stats
        return stats_T, stats_ST


def get_klr_queries(support: int=10, ngrams: int=2) -> List[str]:
    """
    Identify and rank keywords within target and non-target sets.
    Only return the top 10 queries as the precision drops of quickly for the
    niche topic of interest we are looking for (Figure A1 in KLR paper).

    Parameters
    ----------
    """


def _weighted_zone_scoring(
        zone1: np.ndarray, zone2: np.ndarray, weight: float,
    ) -> np.ndarray:
    return weight * zone1 + (1 - weight) * zone2


def _term_frequency(dt_matrix: np.ndarray, mode: str='corpus') -> np.ndarray:
    """
    Compute the frequency of a term in the whole corpus or one document.
    Note that frequency is synonymous with probability.

    Parameters
    ----------
    dt_matrix: Document-term matrix
    mode: Argument to indicate whether the frequency of a term should be computed
        within each document or across all documents, therefore in the whole corpus.
    """
    if mode == 'corpus':
        # Compute term frequency over the whole text corpus.
        term_freq = dt_matrix.sum(0) / dt_matrix.sum()
    if mode == 'document':
        # count terms in each text
        t_count = dt_matrix.sum(axis=1)
        # avoid division by zero
        t_count[t_count==0] = 1
        # Compute term frequency for each document.
        term_freq = dt_matrix / t_count.reshape((-1,1))
    return term_freq


def get_doc_counts_for_term(dt_matrix: sparse.csr_matrix, terms: list) -> Dict[str, int]:
    """
    """
    # count documents containing term
    d_count = dt_matrix.A.sum(axis=0)
    return {term: d_count[index] for index, term in enumerate(terms)}


def _inverse_document_frequency(dt_matrix: np.ndarray) -> np.ndarray:
    """
    Inverse document frequency smooth.

    Parameters
    ----------
    dt_matrix: Document-term matrix
    """
    n_doc = dt_matrix.shape[0]
    n_doc_with_t = np.count_nonzero(dt_matrix, axis=0)
    return np.log(n_doc / (n_doc_with_t + 1)) + 1
