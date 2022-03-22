import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.request import urlopen


directory_project = str(Path(os.path.abspath(__file__)).parent.parent)
logging.basicConfig(
    filename=directory_project + "/ingress.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logger = logging.getLogger(__name__)


def from_url(url: Union[List[str], str], merge: bool=False) -> str:
    if isinstance(url, str):
        try:
            text = str(urlopen(url).read().lower(), "utf-8")
        except Exception:
            logger.info(f"{url} returned with Error")
            text = ""
        return text

    elif isinstance(url, list):
        texts = [from_url(_url) for _url in url]
        if merge:
            texts = (" ").join(texts)
        return texts


def from_file(file_path: str, return_lines: bool=True) -> Union[List[str], str]:
    if return_lines:
        lines = []
        with open(file_path) as f:
            lines = f.readlines()
        return lines
    else:
        return open(file_path, 'rb').read()
