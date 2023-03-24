
import os
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

    chunk_size = 1000
    chunk_overlap = 200

    def load(self,target):

        if os.path.exists(QA_VECTORSTORE_FILE):
            with open(QA_VECTORSTORE_FILE, "rb") as vector_file:
                vectorstore = pickle.load(vector_file)
                #vectorstore.add_documents(documents)
        else:
            loader_cls = UnstructuredPDFLoader
            loader = DirectoryLoader(target, glob="**/[!.]*.pdf", loader_cls=loader_cls, silent_errors=True)

            raw_documents = loader.load()
            print(raw_documents)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            documents = text_splitter.split_documents(raw_documents)
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(documents, embeddings)

            with open(QA_VECTORSTORE_FILE, "wb") as vector_file:
                pickle.dump(vectorstore, vector_file)

        return vectorstore
