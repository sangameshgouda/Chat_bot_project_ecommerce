import chromadb 
from chromadb.utils import embedding_functions
from groq import Groq
import pandas as pd
from dotenv import load_dotenv

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

path  = "/Users/sangameshgoudahorapeti/Documents/GenAI/project-2/Chat_bot_project_ecommerce/app/resources/faq_data.csv"


# Initialize the Groq client
groq_client = Groq()
# Initialize the embedding function using Groq

ef =embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the ChromaDB client
chroma_client = chromadb.Client()

collectiion_name="faq_collection"

def ingestion_data(path):
    if collectiion_name not in [f.name for f in chroma_client.list_collections()]:
        # Create a new collection if it doesn't exist
        chroma_client.create_collection(
            name=collectiion_name, embedding_function=ef)
        df=pd.read_csv(path)
        docs= df['question'].tolist()
        metadatas = [{"answer":i} for i in df['answer'].tolist()]       
        ids = [f"id_{i}" for i in range(len(docs))] 
        chroma_client.get_collection(collectiion_name).add(
            documents=docs, 
            metadatas=metadatas, 
            ids=ids
        )
        print(f"Data ingested into collection: {collectiion_name}")
    else:
        print(f"Collection {collectiion_name} already exists. Skipping ingestion.")


def query_faq(question):
    collection = chroma_client.get_collection(collectiion_name)
    results = collection.query(
        query_texts=[question],
        n_results=1
    )
    return results


def generate_answer(query,context):
    Prompt = '''Given the following context and question, generate answer based on this context only.
    If the answer is not found in the context, kindly state "I don't know". Don't try to make up an answer.
    
    CONTEXT: {context}
    
    QUESTION: {query}'''

    completion = groq_client.chat.completions.create(
        model="llama3-70b-versatile",
        messages=[{"role": "user", "content": Prompt}],
        temperature=0.2)
    return completion.choices[0].message.content


def faq_chain(question):
    results = query_faq(question)
    context = "".join([r.get("answer") for r in results['metadatas'][0]])
    print(f"Context: {context}")
    answer = generate_answer(question, context)
    return answer


if __name__ == "__main__":
    ingestion_data(path)
    question = "What is the return policy?"
    answer = faq_chain(question)
    print(f"Answer: {answer}")
                      
                      


