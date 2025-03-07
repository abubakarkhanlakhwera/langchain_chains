from langchain_google_genai import ChatGoogleGenerativeAI as ChatGoogleGenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")



model = ChatGoogleGenAI(model='gemini-1.5-pro', api_key=api_key,temperature=0.0,max_completion_tokens=100)

# st.header('Research Tool')
# user_input = st.text_input('Enter your prompt')
# if st.button('Summarize'):
#     result = model.invoke(user_input)
#     st.text(result.content)

st.header('Research Tool')
issue_input = st.text_input('Enter the psychological issue the patient is struggling with')


template = PromptTemplate(
    template='''
    You are a compassionate AI designed to provide emotional support through poetic verses. A patient is struggling with the following psychological issue: "{issue_input}". 

    Please generate a short, soothing poem or literary verse that:
    - Acknowledges their emotions with empathy.
    - Provides comfort and hope.
    - Encourages resilience and healing.
    
    The poem should be elegant, heartfelt, and uplifting, leaving the patient feeling heard and reassured.
    ''',
    input_variables=['issue_input']
)
template.save('template_emotional.json')
prompt = template.invoke({
    'issue_input': issue_input
})

if st.button('Generate Explanation'):
    result = model.invoke(prompt)
    st.write(result.content)