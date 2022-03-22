import os
from itertools import compress
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from sdo_and_cc.ingress import RFCFinal, PolyFiles, TextFile
import sdo_and_cc.nlp.concept_extractor as CE
from sdo_and_cc.nlp.utils import (
    text_preprocessing,
    get_diff_of_sets,
    create_training_and_test_sets,
)

folder_data = "/home/christovis/InternetGov/proj2_ietf_and_cc/ietf_and_climate_impact/data"

reference_set = PolyFiles(
    folder_path=folder_data + "/reference_set",
    file_dsc="*.txt",
)
search_set = RFCFinal(
    folder_path=folder_data + "/search_set",
    file_dsc="rfcfinal_%d.txt",
)

reference_set = {
    list(doc.keys())[0]: text_preprocessing(list(doc.values())[0], return_tokens=False)
    for doc in reference_set
}
search_set = {
    did: text_preprocessing(dtext, return_tokens=False)
    for did, dtext in search_set[0].items()
}


y_train, X_train, X_test = create_training_and_test_sets(
    reference_set,
    search_set,
    ref_frac=.33,
    search_frac=.33,
)
print(y_train)
