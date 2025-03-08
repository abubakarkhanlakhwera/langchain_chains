# from pydantic import BaseModel,EmailStr,Field
# from typing import Optional
# # example one
# class Student(BaseModel):
#     name: str
#     age: Optional[int] = None
#     email: EmailStr
#     cgpa: float = Field(gt=0, lt=4.0, default=2.0)
    
# new_student = {'name': 'Joe', 'email': 'abc@hi.com'}

# student = Student(**new_student)
# student_dict = dict(student)
# print(student_dict)
# student_json = student.model_dump_json()
# print(student_json)

# example two
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field

model = ChatOpenAI(max_completion_tokens=50)

class Review(TypedDict):
    key_themes: list[str] = Field(description='Write down all the key themes of the review')
    summary: str = Field(description='The brief summary of the review')
    sentiment: Literal['pos','neg'] = Field(description='Return sentiment of the review either positive, negative or neutral')
    pros: Optional[list[str]] = Field(default=None ,description='List of pros of the review')
    cons: Optional[list[str]] = Field(default=None ,description='List of cons of the review')
    name: Optional[str] = Field(default=None ,description='Write the name of the reviewer')
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke('''
    I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3
processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily
lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me
away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x
actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with
bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard
pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons :
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors

I
''')

print(result)