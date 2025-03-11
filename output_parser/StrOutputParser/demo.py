from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


# Load API key
load_dotenv()


# Initialize Hugging Face endpoint
llm = ChatOpenAI(max_completion_tokens=50)

# Create prompt templates
template1 = PromptTemplate(
    template='Write a detailed report on topic: {topic}',
    input_variables=['topic']
)
template2 = PromptTemplate(
    template='Write a 5-line summary on the following text: \n{text}',
    input_variables=['text']
)

# Generate responses
prompt1 = template1.format(topic='black holes')
result1 = llm.invoke(prompt1)

prompt2 = template2.format(text=result1)
result2 = llm.invoke(prompt2)

print(result2)
