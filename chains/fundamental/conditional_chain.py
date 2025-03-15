from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain.schema.runnable import RunnableParallel ,RunnableBranch ,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class Sentiment(BaseModel):
    sentiment: Literal['positive','negative','neutral'] = Field(description='The sentiment of the text')
str_parser = StrOutputParser()    
parser = PydanticOutputParser(pydantic_object=Sentiment)

prompt1 = PromptTemplate(
    template='Analyze the sentiment of the text: \n {text} and this is \n {format_instruction}',
    input_variables=['text'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

classifier_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive',prompt2 | model | str_parser),
    (lambda x: x.sentiment == 'negative',prompt3 | model | str_parser),
    RunnableLambda(lambda x: 'could not find sentiment')
    
)

result = classifier_chain | branch_chain
print(result.invoke({'text':'I love the product'}))
result.get_graph().print_ascii()


