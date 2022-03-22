import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from tgpp.ingress import TextFile, PDFFile

folder_project = str(Path(os.path.abspath(__file__)).parent.parent.parent.parent)
folder_data = folder_project + "/data"
logger = logging.getLogger(__name__)


class PolyFiles():
    """
    """

    def __init__(
        self,
        folder_path: Optional[str]=None,
        file_dsc: str="*.txt",
        url: Optional[Union[List[str], str]]=None,
    ):
        self.url = url
        self.folder_path = folder_path
        if url:
            self.n_of_docs = len(url)
        else:
            self.file_dsc = file_dsc
            self.file_path = glob.glob(f"{folder_path}/{file_dsc}")
            self.n_of_docs = len(self.file_path)

    def __len__(self) -> int:
        """Get number of documents."""
        return len(self.n_of_docs)

    def __iter__(self):
        """Iterate over all documents."""
        for index in range(0, self.n_of_docs, 1):
            yield self[index]

    def __getitem__(self, index):
        """Get specific document at position `index` within the list."""
        assert 0 <= index < self.n_of_docs, "Index out of bounds."
        if self.url:
            return self.from_url(self.url[index])
        elif self.folder_path:
            return self.from_file(self.file_path[index])

    def from_url(self, url: str) -> Dict[str, str]:
        """
        Returns
        -------
        A dictionary of the form {'url': 'document text', ...}
        """
        if url.endswith('.txt'):
            text = TextFile.from_url(url)
        elif url.endswith('.pdf'):
            text = PDFFile.from_url(url)
        else:
            text = ""
            logger.info(f"Unkown file type for {url}.")
        document_name = urlparse(url).netloc.split('.')[1]
        document = {document_name: text}
        return document

    def from_file(self, file_path: str) -> Dict[str, str]:
        """
        Returns
        -------
        A dictionary of the form {'document id': 'document text', ...}
        """
        if file_path.endswith('.txt'):
            text = TextFile.from_file(file_path)
        elif file_path.endswith('.pdf'):
            text = PDFFile.from_file(file_path)
        else:
            text = ""
            logger.info(f"Unkown file type for {file_path}.")
        document_name = file_path.split('/')[-1]
        document = {document_name: text}
        return document

    def save_to_text_file(self, folder_path: str):
        """
        Download all documents and save them to a text file in `folder_path`.
        """
        if self.url:
            for index in range(0, self.n_of_docs, 1):
                file_name = urlparse(self.url[index]).netloc.split('.')[1]
                with open(f"{folder_path}/{file_name}_{index}.txt", 'w') as file:
                    file.write(list(self[index].values())[0])
        else:
            assert 1 == 2, "Couldn't come up with a reason yet to save file locally that already exists locally."
