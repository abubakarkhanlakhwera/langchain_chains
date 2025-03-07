from langchain_google_genai import ChatGoogleGenerativeAI as ChatGoogleGenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenAI(model='gemini-1.5-pro',api_key=api_key,temperature=0.0,max_completion_tokens=100)

st.title("🩺 Homeopathy Chatbot")

user_symptoms = st.text_input('Enter the symptoms reported by the user')
medical_history = st.text_input('Enter the user\'s medical history')
age = st.number_input('Enter the user\'s age', min_value=3, max_value=120)
gender = st.selectbox('Select your gender',['Male','Female'])
other_details = st.text_input('Enter any additional details about the user')




prompt = PromptTemplate(
    input_variables=["user_symptoms", "medical_history", "age", "gender", "other_details"],
    template="""
    You are an expert homeopathy doctor with deep knowledge of alternative medicine and natural remedies. 
    Your task is to analyze the user's symptoms and suggest appropriate homeopathic remedies.
    
    **User Details:**
    - Age: {age}
    - Gender: {gender}
    - Medical History: {medical_history}
    - Reported Symptoms: {user_symptoms}
    - Additional Information: {other_details}

    Based on the above details, suggest the best homeopathic remedy, including:
    - Remedy Name
    - Potency (e.g., 30C, 200C)
    - Dosage Instructions
    - Possible Side Effects (if any)
    - Lifestyle Tips to Support Healing

    Ensure that your response is in simple language, medically accurate, and suitable for self-care. 
    If the symptoms are severe, recommend consulting a professional homeopath.
    """
)


if st.button('Generate Homeopathy Remedy'):
    chain = prompt | model
    result = chain.invoke({
        'user_symptoms': user_symptoms,
        'medical_history': medical_history,
        'age': age,
        'gender': gender,
        'other_details': other_details    
    }
    )
    st.write(result.content)
else:
    st.write('Please enter the user details and click the button to generate a homeopathy remedy.')