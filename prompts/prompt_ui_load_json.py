from langchain_google_genai import ChatGoogleGenerativeAI as ChatGoogleGenAI
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")



model = ChatGoogleGenAI(model='gemini-1.5-pro', api_key=api_key,temperature=0.0,max_completion_tokens=100)

st.header('Research Tool')
paper_input = st.selectbox( "Select Research Paper Name", ["Select ... ", "Attention Is AllYou Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Modelsare Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical",
"Code-Oriented", "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium(3-5 paragraphs)", "Long (detailed explanation)"] )   

template = load_prompt('template.json')

prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})


if st.button('Generate Explanation'):
    result = model.invoke(prompt)
    st.write(result.content)