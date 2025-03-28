from langchain_openai import ChatOpenAI as ChatLLM
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import (
    RunnableSequence,
    RunnableBranch,
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel
)
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class UserQuery(BaseModel):
    query: str = Field(description="The user's query.")
    classification: str = Field(
        description="Classification of the query. One of: ['complaint', 'refund', 'suggestion']."
    )

model = ChatLLM(temperature=0)
parser = PydanticOutputParser(pydantic_object=UserQuery)
str_parser = StrOutputParser()

template_user_query = PromptTemplate(
    template=(
        "You are a customer support assistant. Classify the user's query into one of the following categories: "
        "[complaint, refund, suggestion].\n\n"
        "Query: {query}\n\n"
        "{format_instruction}"
    ),
    input_variables=["query"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

template_complaint = PromptTemplate(
    template="Generate a response to the complaint: {query}",
    input_variables=["query"]
)

template_refund = PromptTemplate(
    template="Generate a response to the refund request: {query}",
    input_variables=["query"]
)

template_suggestion = PromptTemplate(
    template="Generate a response to the suggestion: {query}",
    input_variables=["query"]
)

classifier_chain = RunnableSequence(template_user_query, model, parser)

def complaint(user_query):
    return user_query.classification == "complaint"

def refund(user_query):
    return user_query.classification == "refund"

def suggestion(user_query):
    return user_query.classification == "suggestion"

branch_chain = RunnableBranch(
    (RunnableLambda(complaint), template_complaint | model | str_parser),
    (RunnableLambda(refund), template_refund | model | str_parser),
    (RunnableLambda(suggestion), template_suggestion | model | str_parser),
    RunnableLambda(lambda x: "Could not classify the query.")
)

result = classifier_chain | branch_chain
result = result.invoke({"query": "You product will be better if you add more features."})
print(result)

