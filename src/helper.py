from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv


def load_data(data):
    loader=DirectoryLoader(data,
                           glob="*.pdf",
                           loader_cls=PyPDFLoader)
    
    docs=loader.load()
    
    return docs

def creating_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    
    return text_chunks

def initialize_embeddings():
    load_dotenv()
    embeddings=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    
    return embeddings