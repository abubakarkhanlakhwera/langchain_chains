from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model='gpt-4', api_key=api_key,temperature=0.5,max_completion_tokens=100)
result = model.invoke('What is capital of India?')
print(result.content)