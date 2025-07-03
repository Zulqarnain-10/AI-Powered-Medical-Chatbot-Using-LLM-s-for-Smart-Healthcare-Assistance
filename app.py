# Importing required libraries and modules
from flask import Flask, render_template, request # Flask web framework and rendering HTML templates, handling HTTP requests
from src.helper import download_hugging_face_embeddings # Custom function to download/load Hugging Face embeddings
from langchain_pinecone import PineconeVectorStore # Pinecone integration for vector storage and search in LangChain
from langchain_openai import OpenAI # OpenAI LLM wrapper for LangChain
from langchain.chains import create_retrieval_chain # Utility to build retrieval-augmented generation chains
from langchain.chains.combine_documents import create_stuff_documents_chain # Utility to combine retrieved docs with LLM
from langchain_core.prompts import ChatPromptTemplate  # Utility to define structured prompt templates
from dotenv import load_dotenv # For loading environment variables from a .env file
from src.prompt import * # Importing custom prompt templates (like system_prompt) from local module
import os # Module for interacting with the operating system (env vars, etc.)

# Initializing Flask web application
app = Flask(__name__)

# Loading environment variables from the .env file
load_dotenv()

# Retrieving API keys from environment
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Setting environment variables programmatically (for redundancy/safety)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initializing Hugging Face embedding model
embeddings = download_hugging_face_embeddings()

# Defining the index name used in Pinecone
index_name = "medicalbot"

# Creating Pinecone vector store using the existing index and embedding model
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Creating a retriever object to fetch top 3 relevant documents based on similarity
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initializing the OpenAI language model (LLM)
llm = OpenAI(
    temperature=0.4,
    max_tokens=500
)

# Creating a prompt template for the conversation using system and user inputs
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),  # Loaded from src/prompt.py
    ("human", "{input}")
])

# Creating a document-processing chain to combine retrieved docs with LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Creating a retrieval-augmented generation (RAG) chain by linking retriever and QA chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Home route – loads chat frontend
@app.route("/")
def index():
    return render_template("chat.html")

# Chat route – handles user input and returns model response
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]  # Get user input from form
    print("User Input:", msg)

    # Get response from LangChain RAG pipeline
    response = rag_chain.invoke({"input": msg})
    clean_answer = response["answer"]

    # Remove "System: " prefix if present (case-insensitive)
    if clean_answer.lower().startswith("system:"):
        clean_answer = clean_answer[len("system:"):].strip()

    print("Response:", clean_answer)
    return clean_answer

# Run Flask server on localhost
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
