from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load API key
load_dotenv()


# Initialize Hugging Face endpoint
model  = ChatOpenAI(max_completion_tokens=50)

# Create prompt templates
template1 = PromptTemplate(
    template='Write a detailed report on topic: {topic}',
    input_variables=['topic']
)
template2 = PromptTemplate(
    template='Write a 5-line summary on the following text: \n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black holes'})

print(result)
