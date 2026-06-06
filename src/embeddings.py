import os

import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY не найден ни в .env, ни в Streamlit Secrets"
        )

    return api_key


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=get_api_key(),
    )


def create_vectorstore(docs):
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        docs,
        embeddings,
    )

    db_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "faiss_index",
    )

    os.makedirs(db_path, exist_ok=True)

    vectorstore.save_local(db_path)

    return vectorstore


def load_vectorstore():
    db_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "faiss_index",
    )

    if not os.path.exists(db_path):
        return None

    embeddings = get_embeddings()

    return FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )