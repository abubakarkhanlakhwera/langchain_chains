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
paper_input = st.selectbox( "Select Research Paper Name", ["Select ... ", "Attention Is AllYou Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Modelsare Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical",
"Code-Oriented", "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium(3-5 paragraphs)", "Long (detailed explanation)"] )   

template = PromptTemplate(template='''
                          Please summarize the research paper titled "{paper_input}" with the following
specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
- Include relevant mathematical equations if present in the paper.
- Explain the mathematical concepts using simple, intuitive code snippets
where applicable.
2. Analogies:
- Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient
information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and
length.
                          ''',
                          input_variables=['paper_input', 'style_input', 'length_input'])


if st.button('Generate Explanation'):
    chain = template | model
    
    result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
    })


    st.write(result.content)