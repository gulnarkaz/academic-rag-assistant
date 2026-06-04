# Academic RAG Assistant

An AI-powered academic research assistant built with **Streamlit**, **LangChain**, and the official **Google Gemini SDK**. This application allows users to upload academic PDF documents, process them into a local vector database using **FAISS**, and perform precise, context-aware Retrieval-Augmented Generation (RAG) to answer research questions without hallucinations.

## 🚀 Features

*   **PDF Document Ingestion:** Upload and parse complex academic papers and PDFs.
*   **Local Vector Storage:** Efficiently chunk and store text embeddings locally using FAISS.
*   **Direct Gemini Integration:** Utilizes the latest stable `gemini-2.5-flash` model via the official Google Generative AI SDK for ultra-fast and reliable academic synthesis.
*   **Strict Academic Guardrails:** The assistant is explicitly prompted to answer *only* based on the provided document context, preventing AI hallucinations.
*   **Clean Streamlit UI:** User-friendly web interface designed for seamless research workflows.

---

## 🛠️ Tech Stack

*   **Frontend & Framework:** Streamlit
*   **Orchestration & Vectorization:** LangChain / LangChain Community
*   **Vector Database:** FAISS (CPU version)
*   **LLM API:** Google Generative AI (`google-generativeai`)
*   **Model:** Gemini 2.5 Flash

---

## 💻 Installation & Local Setup

Follow these steps to clone and run the project locally on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/gulnarkaz/academic-rag-assistant.git](https://github.com/YOUR_USERNAME/academic-rag-assistant.git)
cd academic-rag-assistant

### 2. Set Up a Virtual Environment
Create and activate a clean Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate 
```
### 3. Install Dependencies
Install all required libraries, including the official Google SDK and FAISS:
```bash 
pip install --upgrade pip
pip install streamlit langchain langchain-community langchain-huggingface sentence-transformers pypdf faiss-cpu google-generativeai python-dotenv
```
### 4. Configure Environment Variables
Create a .env file in the root directory of the project and add your Google Gemini API key:
GEMINI_API_KEY=your_actual_api_key_here
### 5. Configure Streamlit (Optional but Recommended)
To prevent internal library warnings from cluttering your terminal, create a .streamlit/config.toml file:
```Ini, TOML
[server]
fastRerun = true

[runner]
magicEnabled = false
```
### 6. Run the Application
Start the Streamlit app:
```bash
./venv/bin/streamlit run src/app.py
```
Once started, open your browser and navigate to:
http://localhost:8501
Upload your academic PDF paper using the sidebar or file uploader.

Initialize/Create the vector database.

Type your research question into the chat input and receive precise answers backed strictly by your document's context!
---
## 📂 Project Structure
```text
academic-rag-assistant/
├── .gitignore               # Specifies intentionally untracked files to ignore (e.g., venv, secrets)
├── .env                     # Local environment variables (GEMINI_API_KEY) - DO NOT COMMIT TO GIT
├── README.md                # Project documentation and setup guide
├── requirements.txt         # List of Python package dependencies
├── .streamlit/
│   └── config.toml          # Streamlit server and performance configuration
├── data/
│   └── faiss_index/         # Local vector database storage (generated automatically)
│       ├── index.faiss
│       └── index.pkl
└── src/
    ├── __init__.py          # Makes src a Python package
    ├── app.py               # Streamlit frontend UI and main application flow
    ├── embeddings.py        # Logic for parsing PDFs and creating local FAISS vector stores
    └── retriever.py         # Custom RAG chain integrated with the official Google Gemini SDK
```
---
📝 License
This project is open-source and available under the MIT License.




