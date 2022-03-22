import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import requests
import PyPDF2
from io import BytesIO

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
        sauce = requests.get(url)
        if sauce.status_code == 200:
            soup = sauce.content
            with BytesIO(soup) as data:
                return _read_pdf(data, url)
        elif sauce.status_code == 404:
            logger.info(f"{url} returned with 404 Error")
            return ""
        else:
            logger.info(f"{url} returned with Error")
            return ""

    elif isinstance(url, list):
        texts = [from_url(_url) for _url in url]
        if merge:
            texts = (" ").join(texts)
        return texts


def from_file(file_path: Union[List[str], str], merge: bool=False) -> List[str]:
    if isinstance(file_path, str):
        text = open(file_path, 'rb')
        return _read_pdf(text, file_path)

    elif isinstance(file_path, list):
        texts = [from_file(_fp) for _fp in file_path]
        if merge:
            texts = (" ").join(texts)
        return texts


def _read_pdf(data: str, source: str) -> str:
    text = []
    pdf = PyPDF2.PdfFileReader(data)
    for page_nr, page in enumerate(range(pdf.getNumPages())):
        try:
            text.append(pdf.getPage(page).extractText())
        except Exception:
            logger.info(f"For {source} page number {page_nr} couldn't be read")
            continue
    return (" ").join(text)

