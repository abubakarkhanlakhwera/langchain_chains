from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("HF_TOKEN")
llm = HuggingFaceEndpoint(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                          task='text-generation')

model = ChatHuggingFace(llm=llm, api_key=api_key,temperature=0.0,max_completion_tokens=10)
result = model.invoke('What is the second largest city in world?')
print(result.content)