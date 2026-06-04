import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings():
    """Инициализирует отличную локальную и бесплатную модель эмбеддингов"""
    # Эта модель от Microsoft прекрасно справляется с техническими и научными текстами
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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