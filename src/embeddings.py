import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

load_dotenv()

# Настраиваем ключ (в облаке он подтянется из Secrets)
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

class DirectGoogleEmbeddings(Embeddings):
    """Кастомный класс эмбеддингов, который делает запросы напрямую через официальный Google SDK, избегая ошибок 404 в LangChain"""
    def embed_documents(self, texts):
        if not texts:
            return []
        # Вызываем стабильный метод через официальный SDK
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']

    def embed_query(self, text):
        if not text:
            return []
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

def get_embeddings():
    """Возвращает наш кастомный, ультра-надежный класс для работы с эмбеддингами"""
    return DirectGoogleEmbeddings()

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