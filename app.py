from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_pinecone import PineconeVectorStore
from flask import Flask, render_template, jsonify, request
from src.helper import initialize_embeddings
from src.prompt import *
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Initialize embeddings
embeddings = initialize_embeddings()

# Pinecone setup
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "llama2"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Prompt template
# First stage
MAP_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# Second stage for map_reduce
REDUCE_PROMPT = PromptTemplate(
    input_variables=["summaries", "question"],
    template="Given the following summaries: {summaries}\nAnswer the question: {question}"
)

chain_type_kwargs = {
    "question_prompt": MAP_PROMPT,
    "combine_prompt": REDUCE_PROMPT
}

# Load local LLaMA 2 model
llm = CTransformers(
    model="E:/GEN AI/End2End-Medical-Chatbot-using-Llama2/models/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    max_new_tokens=128,
    temperature=0.5
)

# Create QA chain
qa_model = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    chain_type="map_reduce"
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"})
    
    print(f"User: {user_message}")
    result = qa_model.invoke({"query": user_message})
    bot_reply = result["result"]
    print("Bot:", bot_reply)
    return jsonify({"reply": bot_reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
