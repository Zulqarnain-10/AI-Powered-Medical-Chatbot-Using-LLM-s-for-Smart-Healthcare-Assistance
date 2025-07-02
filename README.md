# AI-Powered Medical Chatbot Using LLMs for Smart Healthcare Assistance

🚀 **AI-Powered Medical Chatbot** leverages advanced Large Language Models (LLMs) to provide real-time medical query responses and smart healthcare assistance. This project incorporates natural language processing (NLP), semantic search, and vector embeddings to deliver accurate and context-aware answers to users' healthcare-related questions.

> **Disclaimer:**  
> This chatbot is for informational purposes only and is not intended to replace professional medical advice, diagnosis, or treatment.

---

## 🌟 Features

- **AI-Powered Responses:** Utilizes OpenAI GPT models for understanding and responding to natural language medical queries.
- **Semantic Search:** Integrates vector embeddings and semantic similarity (via Pinecone) to retrieve the most relevant medical information.
- **Real-Time Answers:** Fast, cloud-deployed responses for user convenience.
- **User-Friendly Interface:** Built with Flask, HTML, and CSS for an accessible web experience.
- **Scalable & Cloud-Ready:** Deployable on AWS for production-scale use.
- **Secure & Private:** Designed with user privacy in mind.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **AI/ML:** OpenAI GPT, LangChain, Pinecone (Vector Database)
- **Frontend:** HTML, CSS
- **Deployment:** AWS Cloud
- **Additional:** Jupyter Notebook for prototyping and data analysis

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Zulqarnain-10/AI-Powered-Medical-Chatbot-Using-LLM-s-for-Smart-Healthcare-Assistance.git
cd AI-Powered-Medical-Chatbot-Using-LLM-s-for-Smart-Healthcare-Assistance
```

### 2. Set Up the Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory and add your API keys and configuration:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
FLASK_SECRET_KEY=your_secret_key
```

### 4. Run the Application

```bash
flask run
```
The chatbot will be available at `http://127.0.0.1:5000/`.

---

## 🖥️ Project Structure

```
.
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates
├── static/                 # CSS and static files
├── chatbot/                # Core chatbot logic
├── notebooks/              # Jupyter Notebooks for prototyping
└── README.md
```

---

## 🧠 How It Works

1. **User Query:** The user submits a medical question via the web interface.
2. **Semantic Search:** The system uses vector embeddings (Pinecone) to fetch relevant context from medical knowledge bases.
3. **LLM Response:** OpenAI GPT (via LangChain) generates an informative and context-aware answer.
4. **Response Delivery:** The answer is displayed to the user in real-time.

---

## 📦 Deployment

The application can be easily deployed on AWS (EC2, Elastic Beanstalk, or similar). Ensure your environment variables are securely set in your deployment environment.

---

## 🛡️ Disclaimer

This chatbot is **not a substitute for professional medical advice**. Always consult with a qualified healthcare provider for diagnosis and treatment.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📫 Contact

For questions or suggestions, please open an issue or contact [Zulqarnain-10](https://github.com/Zulqarnain-10).
