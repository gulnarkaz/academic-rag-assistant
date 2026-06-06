import os

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY не найден")

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        model_name="gemini-2.5-flash"
    )


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


class DirectRAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.model = get_llm()

    def invoke(self, question):
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        docs = retriever.invoke(question)

        context = format_docs(docs)

        prompt = f"""
Ты академический исследовательский ассистент.

Используй ТОЛЬКО информацию из контекста.

Если ответа в контексте нет,
напиши:

"В предоставленных документах нет информации для ответа."

Контекст:

{context}

Вопрос:

{question}

Ответ:
"""

        response = self.model.generate_content(
            prompt
        )

        return response.text


def build_qa_chain(vectorstore):
    return DirectRAGChain(vectorstore)


def ask_question(chain, question):
    return chain.invoke(question)