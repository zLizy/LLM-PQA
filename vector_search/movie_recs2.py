import pymongo
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AIclient = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

client = pymongo.MongoClient("mongodb+srv://arslan:771944972@cluster0.qv2ymat.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.sample_mflix
collection = db.embedded_movies

def generate_embedding(text: str) -> list[float]:

    # response = openai.embeddings.create(
    #     model="text-embedding-ada-002", 
    #     input=text
    # )
    return AIclient.embeddings.create(input = [text], model="text-embedding-ada-002").data[0].embedding

query = "imaginary characters from outer space at war"

results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": generate_embedding(query),
    "path": "plot_embedding",
    "numCandidates": 100,
    "limit": 4,
    "index": "PlotSemanticSearch",
      }}
]);

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')