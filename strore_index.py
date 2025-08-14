from src.helper import load_data, creating_chunks, initialize_embeddings
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

extracted_data=load_data("data/")
text_chunks=creating_chunks(extracted_data)
embeddings=initialize_embeddings()

# Init Pinecone v4
# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

### As we have Already created Pinecone index and store embeddings there in experiment.ipynb
### file, so we are not going to do it same as it is already saved embeddings(as it takes some time)
### so we here just use existing Index of pinecone, If your are gooing to do it first time,
### un-comment below code and run it will create Index and save embeddings there 


# if "llama2" not in pc.list_indexes().names():
#     pc.create_index(
#         name="llama2",
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )

# # Create vector store from texts
# batch_size = 50  # adjust based on your rate limits
# for i in range(0, len(text_chunks), batch_size):
#     batch = text_chunks[i:i+batch_size]
#     PineconeVectorStore.from_texts(
#         [t.page_content for t in batch],
#         embeddings,
#         index_name="llama2"
#     )

# index_name="llama2"
# docsearch=PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )