# Importing required libraries and helper functions
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Loading environment variables from the .env file
load_dotenv()

# Retrieving and setting Pinecone API key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Loading and processing PDF data
extracted_data = load_pdf_file(data='Data/')           # Step 1: Load PDF files from the 'Data/' directory
text_chunks = text_split(extracted_data)               # Step 2: Split long text into smaller chunks
embeddings = download_hugging_face_embeddings()        # Step 3: Initialize the embedding model from Hugging Face

# Initializing Pinecone client with API key
pc = Pinecone(api_key=PINECONE_API_KEY)

# Defining index name
index_name = "medicalbot"

# Creating a new Pinecone index with appropriate dimension and configuration
pc.create_index(
    name=index_name,
    dimension=384,  # Matching the dimension of the embedding model
    metric="cosine",  # Similarity metric used
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Uploading (upserting) embedded document chunks to the Pinecone vector index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,      # The processed chunks of text from the PDF
    index_name=index_name,      # The name of the Pinecone index to store in
    embedding=embeddings        # The embedding model to convert text into vector form
)
