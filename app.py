from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import CTransformers

from flask import Flask, render_template, jsonify, request

from src.helper import initialize_embeddings
from src.prompt import *
import os
from dotenv import load_dotenv
load_dotenv()

app=Flask(__name__)

embeddings=initialize_embeddings()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name="llama2"
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

PROMPT=PromptTemplate(
    input_variables=["context","question"],
    template=prompt_template
)

chain_type_kwargs={"prompt": PROMPT}

llm = CTransformers(
    model="E:/GEN AI/End2End-Medical-Chatbot-using-Llama2/models/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    max_new_tokens=512,
    temperature=0.5
)

qa_model = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    chain_type="stuff"
)