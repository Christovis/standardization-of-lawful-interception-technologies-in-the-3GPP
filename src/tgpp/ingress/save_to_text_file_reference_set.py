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

folder_data = "../../../data"

urls = TextFile.from_file(folder_data + "/url_topic_inside_focus.txt")
reference_set = PolyFiles(url=urls)
reference_set.save_to_text_file(folder_data)
