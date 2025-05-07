# Importing required modules from LangChain for loading, splitting, and embedding documents
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


# Loading PDF files from a specified directory and extracting their content
def load_pdf_file(data):
    """
    Loads all PDF files from the specified directory and returns the documents.

    Args:
        data (str): Path to the folder containing PDF files.

    Returns:
        documents (list): List of documents loaded from the PDFs.
    """
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# Splitting long documents into smaller text chunks for embedding
def text_split(extracted_data):
    """
    Splits the extracted PDF text into smaller overlapping chunks.

    Args:
        extracted_data (list): List of documents to split.

    Returns:
        text_chunks (list): List of text chunks (with overlap).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Initializing and returning the Hugging Face embedding model
def download_hugging_face_embeddings():
    """
    Downloads and returns the Hugging Face embedding model
    used to convert text into vector format.

    Returns:
        embeddings (HuggingFaceEmbeddings): The embedding object.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    return embeddings
