from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---  Core Tool:  Enhanced Web Search with Focus ---
class ExamWebSearch(DuckDuckGo): #inheriting from DuckDuckGo as it's a good foundation
    def __init__(self, exam_subjects, **kwargs):
        super().__init__(**kwargs)
        self.exam_subjects = exam_subjects

    def _search(self, query):
        focused_query = f"{query} AND ({' OR '.join(self.exam_subjects)})"
        return super()._search(focused_query) # calling base class search function but with focused query


# --- Agent for Knowledge Retrieval ---
knowledge_agent = Agent(
    name="Knowledge Retrieval Agent",
    role="Find information pertinent to academic subjects and competitive exams.",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Use a strong model
    tools=[
        ExamWebSearch(exam_subjects=[  # add your specific exam subjects here
            "history", "geography", "civics", "economics", "science", "math", "aptitude", "current affairs", "general knowledge"
        ]),
    ],
    instructions=[
        "Focus on extracting factual information, definitions, formulas, and key concepts.",
        "Prioritize information from reputable educational sources.",
        "Always cite the source of information using the URL.",
        "Answer the question completely even if it involves information from multiple sources"

    ],
    show_tools_calls=True,
    markdown=True,
)


# --- Agent for Logical Reasoning & Problem Solving ---
reasoning_agent = Agent(
    name="Reasoning Agent",
    role="Apply logical reasoning, problem-solving skills, and calculations to answer questions.",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"), # use same model or another one of choice
    tools=[], # No specific tool as reasoning should be in the model's ability
    instructions=[
        "If the question involves a calculation, show each step clearly.",
         "If the question is analytical, use a structured approach by breaking down the question into smaller parts",
        "Provide a clear and concise answer. ",
         "If you cannot answer the question directly, indicate the reason why, and suggest potential alternatives."
    ],
    show_tool_calls=True,
    markdown=True,

)

# --- Combined Exam Agent ---
exam_agent = Agent(
    team=[knowledge_agent, reasoning_agent],
    instructions=[
       "First find information about the question, then use reasoning to answer.",
       "Use table to display information and answer clearly with sources. ",
    ],
    show_tool_calls=True,
    markdown=True,
)


# --- Example Usage ---
exam_agent.print_response(
    "What is the law of demand in economics? and also list factors which can cause a shift in the demand curve?", stream=True
)


exam_agent.print_response(
   " A train leaves station A at 60km/hr at 10:00 AM . Another train leaves station A at 75 km/hr at 11:00 AM in the same direction. When will the second train catch up to the first train? show all the calculations.",stream=True
)