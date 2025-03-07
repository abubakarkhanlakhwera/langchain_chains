from langchain_core.prompts  import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])
chat_history = []

with open('chat_history.txt') as file:
    chat_history.extend(file.readlines())
    
print(chat_history)
    
propmt = chat_template.invoke({'chat_history': chat_history, 'query': 'where is my refund?'})
print(propmt)