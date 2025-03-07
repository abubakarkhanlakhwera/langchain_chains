from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI as ChatGoogleGenAI

chat_template = ChatPromptTemplate([
    # correct way
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Tell me about this topic {topic}')
    # # incorect way
    # SystemMessage(content="You are a helpful '{domain}' expert"),
    # HumanMessage(content='Tell me about this topic"{topic}"'),
])

prompt = chat_template.invoke({
    'domain':'AI',
    'topic': 'block-chain'
})

print(prompt)