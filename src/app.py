import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from ingest import split_documents
from embeddings import create_vectorstore, load_vectorstore
from retriever import build_qa_chain, ask_question

st.set_page_config(page_title="Academic RAG Assistant", layout="wide")
st.title("📚 Academic RAG Assistant (Powered by Gemini)")

# Папка для временного сохранения загруженных PDF
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_papers")
os.makedirs(UPLOAD_DIR, exist_ok=True)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Боковая панель
with st.sidebar:
    st.header("1. Загрузка документов")
    
    # Виджет для перетаскивания файлов мышкой
    uploaded_files = st.file_uploader(
        "Перетащите сюда научные PDF-статьи", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Инициализировать базу знаний") and uploaded_files:
        with st.spinner("Сохранение и индексация документов..."):
            try:
                all_docs = []
                
                # Сохраняем каждый загруженный файл на диск и сразу читаем его
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())
                    
                    # Загружаем текст из сохраненного PDF
                    loader = PyPDFLoader(file_path)
                    all_docs.extend(loader.load())
                
                if not all_docs:
                    st.error("Не удалось извлечь текст из файлов.")
                else:
                    # Нарезаем на куски и отправляем в FAISS через Gemini
                    split_docs = split_documents(all_docs)
                    vectorstore = create_vectorstore(split_docs)
                    st.session_state.qa_chain = build_qa_chain(vectorstore)
                    st.success(f"Успешно обработано документов: {len(uploaded_files)}!")
            except Exception as e:
                st.error(f"Ошибка при создании базы: {e}")

# Главный экран
st.subheader("2. Задай вопрос по загруженным статьям")

# Пытаемся подгрузить старую базу, если она уже была создана ранее
if st.session_state.qa_chain is None:
    existing_vectorstore = load_vectorstore()
    if existing_vectorstore:
        st.session_state.qa_chain = build_qa_chain(existing_vectorstore)

user_question = st.text_input("Введите ваш вопрос здесь (например: Что такое GraphRAG?):")

if user_question:
    if st.session_state.qa_chain is not None:
        with st.spinner("Gemini анализирует статьи..."):
            try:
                answer = ask_question(st.session_state.qa_chain, user_question)
                st.markdown(f"**Ответ ассистента:**\n\n{answer}")
            except Exception as e:
                st.error(f"Ошибка при генерации ответа: {e}")
    else:
        st.warning("Сначала загрузите файлы и нажмите кнопку инициализации слева.")