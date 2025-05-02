# importing Libraries
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


# Loading the Data
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file(data = 'Data/') # Data Location
text_chunks = text_split(extracted_data) # Text Splitter
embeddings = download_hugging_face_embeddings() #Embedding Model


# Pinecone Initizialization
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# Creating Indexes
pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

# Embed each chunk upsurt the embeddings into your Pinecone Index
docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks, # giving text chunks
    index_name = index_name, # giving index names
    embedding = embeddings, # giving the embedding model
)