from langchain_google_genai import ChatGoogleGenerativeAI as ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model='gemini-2.0-flash',
     temperature=1.0
)

prompt1 = PromptTemplate(
    template='Write a joke about this topic: \n {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='First write the joke the folowing and then Explain  joke: \n {text}',
    input_variables=['text']
)
parser = StrOutputParser()

runnable = RunnableSequence(
    prompt1,
    model,
    prompt2,
    model,
    parser
)

print(runnable.invoke({'topic': 'rats'}))