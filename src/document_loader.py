
import pickle
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


class DocumentLoader:

    chunk_size = 1000
    chunk_overlap = 200

    def load(self,target):
        loader_cls = UnstructuredPDFLoader
        loader = DirectoryLoader(target, glob="**/[!.]*.pdf", loader_cls=loader_cls, silent_errors=True)

        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        documents = text_splitter.split_documents(raw_documents)
        text = ""
        for doc in documents:
            text += doc.page_content.replace("\n", " ")

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)

        return vectorstore