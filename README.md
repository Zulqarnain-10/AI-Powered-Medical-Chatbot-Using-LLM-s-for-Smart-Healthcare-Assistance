# MedBot — AI Medical Q&A Assistant (RAG)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-web%20app-000000?logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-vector%20DB-111827)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> A retrieval-augmented chatbot that answers medical questions from a curated medical reference — grounding every reply in retrieved passages to reduce hallucination.

> **Disclaimer:** For informational purposes only. Not a substitute for professional medical advice, diagnosis, or treatment.

## Problem

General LLMs answer medical questions from parametric memory, with no grounding and no traceable source — risky for health information. A safer pattern constrains answers to a trusted medical reference and retrieves relevant passages before generating a response.

## Approach

MedBot is a Retrieval-Augmented Generation (RAG) pipeline over a medical reference text:

- **Ingest & index** — load the source PDF (`Data/`), split into overlapping chunks (`RecursiveCharacterTextSplitter`, 500 / 20), embed with HuggingFace **`all-MiniLM-L6-v2`** (384-dim), and upsert into a **Pinecone** serverless index (cosine similarity). Run once via `store_index.py`.
- **Retrieve & answer** — at query time, embed the question, retrieve the most similar chunks from Pinecone, and pass them as context to an **OpenAI** chat model through LangChain's `create_retrieval_chain`. The system prompt constrains the model to the retrieved context and to reply "I don't know" when the answer isn't supported.
- **Serve** — a **Flask** web UI for real-time chat.

## Results

- Answers are **grounded in retrieved passages** from the medical reference rather than free-form generation, and kept concise (≤ 3 sentences).
- The assistant **declines** when the retrieved context doesn't support an answer, reducing confident hallucination.

> No formal accuracy benchmark is included yet. A labeled evaluation set (precision / recall / F1 over a question bank) is on the roadmap.

## Tech stack

`Python` · `Flask` · `LangChain` · `Pinecone` · `HuggingFace Sentence-Transformers (all-MiniLM-L6-v2)` · `OpenAI API` · `PyPDF`

## Demo

![MedBot chat interface](https://github.com/user-attachments/assets/4969e5fe-2686-4e90-b36d-95be0b7f92b2)
![MedBot answering a query](https://github.com/user-attachments/assets/c8d0e81c-2818-4a2e-af67-625c2f4ce4d6)

## How to run

```bash
# 1. Clone
git clone https://github.com/Zulqarnain-10/AI-Powered-Medical-Chatbot-Using-LLM-s-for-Smart-Healthcare-Assistance.git
cd AI-Powered-Medical-Chatbot-Using-LLM-s-for-Smart-Healthcare-Assistance

# 2. Environment + dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. API keys — create a .env file
#    OPENAI_API_KEY=...
#    PINECONE_API_KEY=...

# 4. Build the vector index (one-time, reads PDFs from Data/)
python store_index.py

# 5. Launch
python app.py
```

Then open the local URL shown in the terminal.

## Roadmap

- Add a labeled evaluation set and report retrieval/answer quality (precision, recall, F1).
- Swap in a larger medical corpus and citations back to source passages.
- Containerize and deploy.

## License

MIT — see [LICENSE](LICENSE).
