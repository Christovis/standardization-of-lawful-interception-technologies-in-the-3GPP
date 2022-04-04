import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.request import urlopen

from tgpp.config.config import CONFIG


def from_url(url: Union[List[str], str], merge: bool=False) -> str:
    if isinstance(url, str):
        try:
            text = str(urlopen(url).read().lower(), "utf-8")
        except Exception:
            text = ""
        return text

    elif isinstance(url, list):
        texts = [from_url(_url) for _url in url]
        if merge:
            texts = (" ").join(texts)
        return texts


def from_file(file_path: str) -> List[str]:
    lines = []
    with open(file_path) as file:
        while (line := file.readline().rstrip()):
            lines.append(line)
    return list(set(lines))
