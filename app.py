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
app = Flask(__name__) # Creates a Flask app instance

# Loading environment variables from the .env file
load_dotenv() # Loads variables like API keys from a .env file into environment

# Retrieving API keys from environment
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') # Fetch Pinecone API key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') # Fetch OpenAI API key

# Fail fast with a named error instead of an opaque TypeError further down
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing PINECONE_API_KEY and/or OPENAI_API_KEY. "
        "Set them as Space secrets (Settings -> Variables and secrets) or in a local .env file."
    )

# Setting environment variables programmatically (for redundancy/safety)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY # Ensure Pinecone API key is set in env
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # Ensure OpenAI API key is set in env

# Initializing Hugging Face embedding model
embeddings = download_hugging_face_embeddings() # Load or download the embeddings model

# Defining the index name used in Pinecone
index_name = "medicalbot" # Name of the Pinecone vector index (pre-existing or to be created)

# Building the index once if it does not exist yet, so a fresh deployment
# (for example a new Hugging Face Space with a fresh Pinecone project) boots
# itself from Data/ without a separate store_index.py run.
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# The vector count, not the index's existence, decides whether to ingest: an
# interrupted first boot leaves the index created but empty, and existence-only
# checks would skip the build forever after.
if pc.Index(index_name).describe_index_stats().total_vector_count == 0:
    from src.helper import load_pdf_file, text_split
    print(f"Pinecone index '{index_name}' is empty - building it from Data/ (one-time, several minutes)...")
    text_chunks = text_split(load_pdf_file(data='Data/'))
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings
    )
    print(f"Pinecone index '{index_name}' built.")

# Creating Pinecone vector store using the existing index and embedding model
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
) # Sets up a vector store for semantic document search

# Creating a retriever object to fetch top 3 relevant documents based on similarity
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
) # Retriever for top 3 most similar documents

# Initializing the OpenAI language model (LLM)
llm = OpenAI(
    temperature=0.4, # Lower temperature for more deterministic responses
    max_tokens=500 # Limit the length of responses
) # Language model for answering questions

# Creating a prompt template for the conversation using system and user inputs
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),  # Loaded from src/prompt.py
    ("human", "{input}") # Placeholder for user input
])

# Creating a document-processing chain to combine retrieved docs with LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# This chain uses the LLM to process and synthesize answers from the retrieved documents

# Creating a retrieval-augmented generation (RAG) chain by linking retriever and QA chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# The main pipeline: retrieve docs, then use LLM to answer based on them

# Home route – loads chat frontend
@app.route("/")
def index():
    return render_template("chat.html") # Renders the chat user interface

# Chat route – handles user input and returns model response
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]  # Get user input from form
    print("User Input:", msg)

    # Get response from LangChain RAG pipeline
    response = rag_chain.invoke({"input": msg}) # Run the full retrieval + LLM pipeline
    clean_answer = response["answer"] # Extract generated answer

    # Remove "System: " prefix if present (case-insensitive)
    if clean_answer.lower().startswith("system:"):
        clean_answer = clean_answer[len("system:"):].strip() # Clean up formatting

    print("Response:", clean_answer)
    return clean_answer # Return the response to the frontend

# Run Flask server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False) # Port from env for hosted deployments
