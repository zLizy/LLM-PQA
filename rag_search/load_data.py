from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import openai
import gradio as gr
from gradio.themes.base import Base
import key_param  
import os 
import shutil

# client = MongoClient('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')
client = MongoClient("mongodb+srv://arslan:771944972@cluster0.qv2ymat.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
dbName = "models"
collectionName = "collection_of_model_description"
collection = client[dbName][collectionName]

# loader = DirectoryLoader('rag_search\model_profile_files',glob="./*.txt",show_progress=True)
# data = loader.load()

# embeddings = OpenAIEmbeddings(openai_api_key=key_param.OPENAI_API_KEY)

# vectorStore = MongoDBAtlasVectorSearch.from_documents(data,embeddings,collection=collection )


def upload_and_move_files():
    loader = DirectoryLoader('D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_profile_files',glob="./*.txt",show_progress=True)
    data = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=key_param.OPENAI_API_KEY)
    vectorStore = MongoDBAtlasVectorSearch.from_documents(data,embeddings,collection=collection )

    source_dir = 'D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_profile_files'
    target_dir = 'D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\saved_model_profiles'

    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)

        # Move the file
        shutil.move(file_path, os.path.join(target_dir, file_name))
        print(f"Moved {file_name} to {target_dir}")

if __name__ == '__main__':
    upload_and_move_files()