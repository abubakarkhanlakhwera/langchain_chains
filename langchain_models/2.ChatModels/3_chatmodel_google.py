from langchain_google_genai import ChatGoogleGenerativeAI as ChatGoogleGenAI
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenAI(model='gemini-1.5-pro', api_key=api_key,temperature=0.0,max_completion_tokens=100)
result = model.invoke('What is capital of pakistan?')
print(result.content)