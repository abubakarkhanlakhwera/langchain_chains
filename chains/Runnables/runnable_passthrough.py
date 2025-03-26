from langchain_groq import ChatGroq as ChatLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

prompt_1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatLLM(
    model='llama-3.3-70b-versatile',
    temperature=1.0
)

parser = StrOutputParser()

prompt_2 = PromptTemplate(
    template='Explain the following joke \n  {joke}',
    input_variables=['joke']
)

joke_gen_chain = RunnableSequence(prompt_1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt_2 , model , parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic': 'cricket game'})

joke = result.get('joke')
explanation = result.get('explanation')

print(f"Joke: {joke}\n\n\n\n ----------------------------------------------------------------------" )
print("Explanation:", explanation)