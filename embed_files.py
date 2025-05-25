import os 
import pandas as pd 
import numpy as np
import PyPDF2   
import openai
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import chromadb
from uuid import uuid4
from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OpenAI_API_Key')


cwd = os.getcwd()

data_path = os.path.join(cwd, 'data')
os.listdir(data_path)
metadata = []
for element in os.listdir(data_path):
    element_path = os.path.join(data_path, element)
    if os.path.isdir(element_path):
        for subelement in os.listdir(element_path):
            subelement_path = os.path.join(element_path, subelement)
            if os.path.isfile(subelement_path) and subelement.endswith('.pdf'):
                print({"department": element, "doc_name": subelement, "path": subelement_path})
                metadata.append({"department": element, "doc_name": subelement, "path": subelement_path})
    else:
        pass


for element in metadata:
    with open(element['path'], 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        element['text'] = text
        element['num_pages'] = len(reader.pages)
        element['num_words'] = len(text.split())
        element['num_chars'] = len(text)
        element['num_sentences'] = text.count('.') + text.count('!') + text.count('?')
        element['num_paragraphs'] = text.count('\n\n') + 1

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

final_chunks = []
for i, element in enumerate(metadata):
    text_chunks = text_splitter.split_text(element['text'])
    final_chunks.append(text_chunks)


def get_chunk_metadata(final_chunks, metadata):
    final_docs = []
    i = 0
    for idx, doc in enumerate(final_chunks):
        for chunk in doc:
            final_docs.append(Document(
                page_content=chunk,
                metadata={ 
                'source' : metadata[idx]['path'],             
                'department': metadata[idx]['department'],
                'doc_name': metadata[idx]['doc_name']     
                },
                id = str(i)
            ))
            i += 1
    return final_docs
final_docs = get_chunk_metadata(final_chunks, metadata)

uuids = [str(uuid4()) for _ in range(len(final_docs))]
#chroma_client = chromadb.PersistentClient()


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


vector_store = Chroma(
    collection_name="knowledge_graph_collection",
    embedding_function=embeddings,
    persist_directory="./ChromaNew",
    collection_metadata={
        "hnsw:space": "cosine",  # Distance metric (cosine, l2, or ip)
        "hnsw:M": 32,           # Number of neighbors for HNSW graph (higher = better accuracy, more memory)
        "hnsw:construction_ef": 100,  # Neighbors explored during index construction
        "hnsw:search_ef": 50,    # Neighbors explored during search
        "hnsw:num_threads": 4,   # Number of threads for HNSW operations
        "hnsw:resize_factor": 1.2,  # Growth rate for graph capacity
        "hnsw:sync_threshold": 1000  # Threshold for syncing index to disk
    } 
)
vector_store.add_documents(documents=final_docs, ids=uuids)

vector_store.persist()

query_results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("What are the model techniques used in satellite imagery?"),
    k=2
   # filter={"department" : "Ops"} # Filter by category
)

query_results2 = vector_store.similarity_search_with_score(
    query="What are the model techniques used in satellite imagery?",
    k=2
    # filter={"department" : "Ops"} # Filter by category        
)
print("Query Results printing ................")
print(query_results)
print("Query Results2 printing .........***.......")
print(query_results2)
#print(vector_store.get(limit=2))
#print(vector_store.get(ids=[uuids[0]]))




