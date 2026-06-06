import os
import traceback
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader

from ingest import split_documents
from embeddings import (
    create_vectorstore,
    load_vectorstore,
)
from retriever import (
    build_qa_chain,
    ask_question,
)

st.set_page_config(
    page_title="Academic RAG Assistant",
    layout="wide",
)

st.title("📚 Academic RAG Assistant")

UPLOAD_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "sample_papers",
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True,
)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

with st.sidebar:
    st.header("Загрузка PDF")

    uploaded_files = st.file_uploader(
        "Выберите PDF-файлы",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Инициализировать базу знаний"):

        if not uploaded_files:
            st.warning(
                "Сначала загрузите PDF-файлы"
            )

        else:
            with st.spinner(
                "Создание векторной базы..."
            ):
                try:
                    all_docs = []

                    for uploaded_file in uploaded_files:

                        file_path = os.path.join(
                            UPLOAD_DIR,
                            uploaded_file.name,
                        )

                        with open(
                            file_path,
                            "wb",
                        ) as f:
                            f.write(
                                uploaded_file.getbuffer()
                            )

                        loader = PyPDFLoader(
                            file_path
                        )

                        all_docs.extend(
                            loader.load()
                        )

                    split_docs = split_documents(
                        all_docs
                    )

                    vectorstore = (
                        create_vectorstore(
                            split_docs
                        )
                    )

                    st.session_state.qa_chain = (
                        build_qa_chain(
                            vectorstore
                        )
                    )

                    st.success(
                        f"Обработано файлов: {len(uploaded_files)}"
                    )

                except Exception as e:
                    st.error(
                        f"Ошибка: {e}"
                    )

                    st.code(
                        traceback.format_exc()
                    )

st.subheader(
    "Задайте вопрос по документам"
)

if st.session_state.qa_chain is None:
    try:
        vectorstore = load_vectorstore()

        if vectorstore:
            st.session_state.qa_chain = (
                build_qa_chain(
                    vectorstore
                )
            )

    except Exception:
        pass

question = st.text_input(
    "Введите вопрос"
)

if question:

    if st.session_state.qa_chain is None:
        st.warning(
            "Сначала создайте базу знаний"
        )

    else:
        with st.spinner(
            "Анализ документов..."
        ):
            try:
                answer = ask_question(
                    st.session_state.qa_chain,
                    question,
                )

                st.markdown(answer)

            except Exception as e:
                st.error(str(e))

                st.code(
                    traceback.format_exc()
                )