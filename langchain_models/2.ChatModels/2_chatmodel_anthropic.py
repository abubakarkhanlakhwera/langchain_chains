from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
model = ChatAnthropic(model='gpt-4', api_key=api_key,temperature=0.5,max_completion_tokens=100)
result = model.invoke('What is capital of Australia?')
print(result.content)
