from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1= PromptTemplate(
    template='generate a random title of a story'
    
)

prompt2 = PromptTemplate(
    template='First write the title of the story and then write a story on this title: \n {title}',
    input_variables=['title']
    
)
model = ChatOpenAI(max_completion_tokens=50)
parser = StrOutputParser()

chain = prompt1 | model | prompt2 | model | parser

result = chain.invoke({})
print(result)
chain.get_graph().print_ascii()