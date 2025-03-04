from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model='gpt-3.5-turbo-instruct', api_key=api_key)
result = llm.invoke('What is capital of India?')
print(result)
 