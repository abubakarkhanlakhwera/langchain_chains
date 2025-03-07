from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    'capital of india is delhi',
    'capital of france is paris',
    'capital of germany is berlin',
    'capital of italy is rome',
    'capital of japan is tokyo',
    'capital of china is beijing',
    'capital of russia is moscow'
]

query = "What is the capital of India?"
result_doc =embeddings.embed_documents(documents)
result_query = embeddings.embed_query(query)

similarity = cosine_similarity([result_query], result_doc)[0]
enumerate_similarity = list(enumerate(similarity)) # add index to similarity
sorted_similarity = (sorted(enumerate_similarity),lambda x: x[1]) # sort by similarity
index, score = sorted_similarity[0][0] 
print('Most similar document:', documents[index])
print('Similarity score:', score)