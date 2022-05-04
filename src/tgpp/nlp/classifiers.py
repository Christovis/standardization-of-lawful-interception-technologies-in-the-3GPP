import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import (
    svm,
    tree,
    naive_bayes,
    linear_model,
    ensemble,
    neighbors,
)

logger = logging.getLogger(__name__)


def Classifiers(
    y_train: List[int],
    X_train: np.ndarray,
    X_test: np.ndarray,
    docnames: List[str],
    algorithms=['nbayes', 'nearest', 'logit', 'SVM', 'LDA', 'tree', 'gboost', 'rf'],
    rf_trees: int=200,
    seed: int=12345,
):
    """
    Notes
    -----
    The classification labels are zero for target set and one otherwise.

    Parameters
    ----------
    y_train, X_train, X_test:
    algorithms:
    rf_trees:
    seed: Seed for random number generator.
    """
    # Get probability of reference set from classifiers
    results = {}
    logger.info("Start classifications")

    # Naive Bayes
    if 'nbayes' in algorithms:
        clf_nb = naive_bayes.MultinomialNB()
        clf_nb.fit(X_train, y_train)
        results['nbayes'] = clf_nb.predict(X_test).tolist()
        logger.info("Finished nbayes classifier")

    # Nearest Neighbor
    if 'nearest' in algorithms:
        clf_nn = neighbors.KNeighborsClassifier()
        clf_nn.fit(X_train, y_train)
        results['nearest'] = clf_nn.predict(X_test).tolist()
        logger.info("Finished NN classifier")

    # Logit
    if 'logit' in algorithms:
        clf_logit = linear_model.LogisticRegression()
        clf_logit.fit(X_train, y_train)
        results['logit'] = clf_logit.predict(X_test).tolist()
        logger.info("Finished logit classifier")

    # Support vector machine
    if 'SVM' in algorithms:
        clf_svm = svm.SVC(C=100, probability=True, random_state=seed)
        clf_svm.fit(X_train, y_train)
        results['svm'] = clf_svm.predict(X_test).tolist()
        logger.info("Finished SVM classifier")

    # Linear discriminant
    if 'LDA' in algorithms:
        clf_lda = LDA()
        clf_lda.fit(X_train.toarray(), y_train)
        results['lda'] = clf_lda.predict(X_test.toarray()).tolist()
        logger.info("Finished LDA classifier")

    # Tree
    if 'tree' in algorithms:
        clf_tree = tree.DecisionTreeClassifier(random_state=seed)
        clf_tree.fit(X_train.toarray(), y_train)
        results['tree'] = clf_tree.predict(X_test.toarray()).tolist()
        logger.info("Finished Tree classifier")

    # Gradient boosting
    if 'gboost' in algorithms:
        clf_gboost = ensemble.GradientBoostingClassifier(random_state=seed)
        clf_gboost.fit(X_train.toarray(), y_train)
        results['gboost'] = clf_gboost.predict(X_test.toarray()).tolist()
        logger.info("Finished Gboost classifier")

    # Random forest
    if 'rf' in algorithms:
        clf_rf = ensemble.RandomForestClassifier(n_estimators=rf_trees, random_state=seed)
        clf_rf.fit(X_train.toarray(), y_train)
        results['rf'] = clf_rf.predict(X_test.toarray()).tolist()
        logger.info("Finished RF classifier")


    # create DataFrame and invert T with S\T labels
    df = 1 - pd.DataFrame(results, index=docnames)
    # sum of all classifier predictions
    target_votecount = df.sum(axis=1)

    # Group documents into T- and S-set depending on classifiers
    docnames_T, docnames_ST = get_target_set(target_votecount, vote_min=1)
    print(f"Classifier: T={len(docnames_T)} ST={len(docnames_ST)}")
    return docnames_T, docnames_ST


def get_target_set(
    target_votecount: pd.DataFrame, vote_min: int=1,
) -> Tuple[List[str], List[str]]:
    """
    Identify an estimated target set within search set using classifier votes.

    Parameters
    ----------
    target_votecount:
    vote_min: Min number of classifiers that assigned document to target set.
    """
    target_docnames = list(target_votecount[target_votecount >= vote_min].index)
    nontarget_docnames = list(target_votecount[target_votecount < vote_min].index)
    return target_docnames, nontarget_docnames
