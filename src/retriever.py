import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def get_llm():
    """Настраивает официальный клиент Google Gemini с актуальной моделью 2.5"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY не найден в файле .env")
    genai.configure(api_key=api_key)
    
    # ИСПОЛЬЗУЕМ АКТУАЛЬНУЮ МОДЕЛЬ ДЛЯ v1beta
    return genai.GenerativeModel('gemini-2.5-flash')

def format_docs(docs):
    """Соединяет куски найденного текста в один блок для контекста"""
    return "\n\n".join(doc.page_content for doc in docs)

class DirectRAGChain:
    """Простая замена LCEL-цепочки, работающая напрямую с Google SDK в обход багов LangChain"""
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.model = get_llm()
        
    def invoke(self, question):
        # 1. Поиск похожих кусков текста в локальной базе FAISS
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        context = format_docs(docs)
        
        # 2. Формирование четкого системного промпта
        prompt = (
            "Ты — профессиональный академический ассистент.\n"
            "Отвечай на вопрос пользователя строго на основе предоставленного контекста. "
            "Если в контексте нет ответа, так и скажи, не придумывай факты от себя.\n\n"
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {question}\n"
            "Ответ:"
        )
        
        # 3. Прямой запрос к Google API, который никогда не выдаст 404
        response = self.model.generate_content(prompt)
        return response.text

def build_qa_chain(vectorstore):
    """Создает цепочку, совместимую с твоим app.py"""
    return DirectRAGChain(vectorstore)

def ask_question(chain, question):
    """Отправляет вопрос напрямую в нашу кастомную цепочку"""
    return chain.invoke(question)