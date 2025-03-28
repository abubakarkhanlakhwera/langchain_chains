from langchain_groq import ChatGroq as ChatLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnablePassthrough,RunnableParallel
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
    return len(text.split())

llm = ChatLLM(
    model='llama-3.3-70b-versatile',
    temperature=1.0
)
parser = StrOutputParser()
template = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(template, llm, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic': 'AI'})

joke = result.get('joke')
word_count = result.get('word_count')
print(f"Joke: {joke}\n\n\n\n ----------------------------------------------------------------------" )
print("Word Count:", word_count)