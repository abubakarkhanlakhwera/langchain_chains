from langchain_google_genai import ChatGoogleGenerativeAI as ChatGoogleGenAI
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenAI(model='gemini-1.5-pro',api_key=api_key,temperature=0.0,max_completion_tokens=100)

chat_history = [
    SystemMessage(content='You are a helpfull asistance'),
    
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print('Chatbot:', result.content)

print(chat_history)