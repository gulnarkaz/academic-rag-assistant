import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "sample_papers",
)


def load_documents():
    documents = []

    if not os.path.exists(DATA_PATH):
        return documents

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file)

            loader = PyPDFLoader(file_path)

            documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    return splitter.split_documents(documents)