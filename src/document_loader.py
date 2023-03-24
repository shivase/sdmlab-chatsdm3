
import os
import logging
import pickle
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from langchain.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
    UnstructuredHTMLLoader,
    ReadTheDocsLoader,
    DirectoryLoader,
)

# Map loader class abbreviations to actual classes
FILE_LOADER_CLASSES = {
    'text': TextLoader,
    'pdf_miner': PDFMinerLoader,
    'pymupdf': PyMuPDFLoader,
    'pdf': UnstructuredPDFLoader,
    'html': UnstructuredHTMLLoader,
}

DIR_LOADER_CLASSES = {
    'directory': DirectoryLoader,
    'readthedocs': ReadTheDocsLoader,
}


load_dotenv()
QA_VECTORSTORE_FILE=os.environ.get("QA_VECTORSTORE_FILE")

class DocumentLoader:

    singleton = None
    chunk_size = 1000
    chunk_overlap = 200
    vectorstore = None

    def __new__( cls, *args, **kwargs ):
        if cls.singleton is None:
            cls.singleton = super().__new__( cls )
        return cls.singleton

    def vectorized(self, loader):
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        documents =  text_splitter.split_documents(raw_documents)
        if self.vectorstore is None:
            embeddings = OpenAIEmbeddings()
            self.vectorstore =  FAISS.from_documents(documents, embeddings)
        else:
            self.vectorstore.add_documents(documents)

    def add_documents(self, file, loader_cls):
        logging.info("Loading documents from %s", file)
        loader = FILE_LOADER_CLASSES[loader_cls](file)
        self.vectorized(loader)
        logging.info("Loading documents from %s finished", file)
        return self.vectorstore

    def load(self,target):

        if os.path.exists(QA_VECTORSTORE_FILE):
            with open(QA_VECTORSTORE_FILE, "rb") as vector_file:
                self.vectorstore = pickle.load(vector_file)
        else:
            loader_cls = UnstructuredPDFLoader
            loader = DirectoryLoader(target, glob="**/[!.]*.pdf", loader_cls=loader_cls, silent_errors=True)

            self.vectorized(loader)

            with open(QA_VECTORSTORE_FILE, "wb") as vector_file:
                pickle.dump(self.vectorstore, vector_file)

        return self.vectorstore
