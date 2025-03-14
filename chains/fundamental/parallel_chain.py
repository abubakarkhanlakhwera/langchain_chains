from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='generate a poem on business and its hardship and must be ironic'
)
prompt2 = PromptTemplate(
    template='Give me the summary of then poem \n the poem is \n {poem}',
    input_variables=['poem']
)
prompt3 = PromptTemplate(
    template='Make some questions on the poem \n the poem is \n {poem}',
    input_variables=['poem']
)
prompt4 = PromptTemplate(
    template="Answer the questions using the summary of the poem:\n\nSummary:\n{summary}\n\nQuestions:\n{questions}\n\nProvide the answers in a structured format, listing the question first, followed by the answer.",
    input_variables=['summary','questions']
)

poem_chain = prompt1 | model | parser

parallel = RunnableParallel({
    'summary' : prompt2 | model | parser,
    'questions' : prompt3 | model | parser
})

linear = prompt4 | model | parser

poem = poem_chain.invoke({})

merged_chain = parallel | linear
result = merged_chain.invoke({'poem':poem})
print(result)
merged_chain.get_graph().print_ascii()