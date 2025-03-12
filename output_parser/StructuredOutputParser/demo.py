from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='google/gemma-2-2b-it',
                          task='text-generation',
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='Fact_1',description='The first fact about the topic',type='str'),
    ResponseSchema(name='Fact_2',description='The second fact about the topic',type='str'),
    ResponseSchema(name='Fact_3',description='The third fact about the topic',type='str'),
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give three facts about the  topic: {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic': 'black holes'})
print(result)