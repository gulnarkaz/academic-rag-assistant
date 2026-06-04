import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings():
    """Инициализирует облачную модель эмбеддингов от Google, экономя память сервера"""
    # API-ключ автоматически подтянется из переменных окружения Streamlit Secrets
    return GoogleGenerativeAIEmbeddings(model="text-embedding-004")

def create_vectorstore(docs):
    """Создает векторную базу FAISS на основе кусков текста"""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Путь для сохранения базы локально в папку data/faiss_index
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    vectorstore.save_local(DB_PATH)
    return vectorstore

def load_vectorstore():
    """Загружает существующую векторную базу с диска"""
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")
    if os.path.exists(DB_PATH):
        embeddings = get_embeddings()
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None