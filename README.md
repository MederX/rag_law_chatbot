# 🇰🇬 RAG Chatbot for Kyrgyz Laws

This is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about the laws of the Kyrgyz Republic. The system uses LangChain, FAISS for vector storage, HuggingFace embeddings, and a local LLM served via Ollama (LLaMA3 3B Instruct).

## 📌 Features

* Answers legal questions based on Kyrgyz laws using contextual retrieval.
* Powered by LLaMA3 3B through Ollama.
* Embeddings generated with SentenceTransformers on CPU or GPU.
* Russian-only legal expert prompt, tuned for clarity and precision.
* Supports FAISS vector index for fast semantic search.

## 🧠 Tech Stack

* [LangChain](https://github.com/langchain-ai/langchain)
* [HuggingFace SentenceTransformers](https://www.sbert.net/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Ollama](https://ollama.com/)
* Python 3.10+

## 🗂️ Project Structure

```
rag_kyrgyz_laws/
├── rag_chatbot_2.py         # Main script for querying the chatbot
├── data/                    # Folder containing Kyrgyz laws in .txt format
├── index/                   # FAISS index and metadata (generated)
├── venv/                    # Optional: your Python virtual environment
└── README.md                # This file
```

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/rag_kyrgyz_laws.git
cd rag_kyrgyz_laws
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install FAISS**

```bash
# For CPU
pip install faiss-cpu

# Or, if using GPU (ensure your PyTorch is built with CUDA)
pip install faiss-gpu
```

5. **Install and run Ollama**

* Download and install Ollama from [https://ollama.com](https://ollama.com)
* Pull and start the LLaMA3 model:

```bash
ollama pull llama3
ollama run llama3
```

## 💾 Usage

1. Prepare your `.txt` files with Kyrgyz legal texts in the `data/` folder.

2. Run the chatbot script:

```bash
python rag_chatbot_2.py
```

3. Ask your legal question in Russian. The model will return a concise answer based strictly on the context from the law.

## 🧠 Example Prompt Template

```
Ты эксперт по законам Кыргызской Республики, который отвечает только на русском языке и ни в коем случае не используй английские слова.
Используй следующий контекст для ответа на вопрос:
{context}
Вопрос:
{question}
Дай точный, краткий и понятный ответ, только на основе этого контекста. Максимум 3-5 предложений.
Ответ:
```

## ✅ Sample Question

> Какое наказание предусмотрено за развратные действия без сексуального контакта в отношении несовершеннолетнего младше 16 лет?

## 🛠️ Troubleshooting

* **CUDA errors**: Make sure your PyTorch installation supports CUDA and matches your GPU driver.
* **FAISS import error**: Install the correct `faiss-cpu` or `faiss-gpu` depending on your setup.
* **Ollama not running**: Ensure the Ollama server is started with the appropriate model.

## 📜 License


