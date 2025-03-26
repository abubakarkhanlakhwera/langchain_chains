from langchain_groq import ChatGroq as ChatLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatLLM(
    model='llama-3.3-70b-versatile',
    temperature=1.0
)
parser = StrOutputParser()

template_X = PromptTemplate(
    template='Make a tweet about this topic: \n {topic}',
    input_variables=['topic']
) 

template_Linkedin = PromptTemplate(
    template='Make a linkedin post about this topic: \n {topic}',
    input_variables=['topic']
)

chain = RunnableParallel({
    'tweet': RunnableSequence(
        template_X,
        model,
        parser
    ),
    'linkedin': RunnableSequence(
        template_Linkedin,
        model,
        parser
    )
})

result = chain.invoke({'topic': 'MCP'})

# Extract and print the tweet and LinkedIn post separately
tweet = result.get('tweet')
linkedin_post = result.get('linkedin')

print(f"Tweet:{tweet}\n\n\n\n ----------------------------------------------------------------------" )
print("LinkedIn Post:", linkedin_post)
