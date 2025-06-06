from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pyobjects.pyobj import State

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

# Step 5: LLM Prompt for Structured Extraction
prompt_template = PromptTemplate.from_template(
    template="""
You are an expert analyst. Based on the following information, extract and return structured company data in **valid JSON**.

Only output JSON, with no explanatory text.

{web_content}

Return the data in this exact JSON format and DO NOT include any other text:
{{
  "Company Name": "",
  "Country": "",
  "Primary Address": "",
  "Registration Number": "",
  "Legal Form": "",
  "Town": "",
  "Email": "",
  "Phone": "",
  "Website": "",
  "General Details": "",
  "Directors & Shareholders": [],
  "UBO (Ultimate Beneficial Owner)": "",
  "Subsidiaries": [],
  "Parent Company": "",
  "Last Reported Revenue": ""
}}
"""
)

structured_chain = prompt_template | model | parser
def extract_data(state: State) -> State:
    data = structured_chain.invoke({
    "web_content": state["web_content"],
    "company_name": state["company_name"],
    "country": state["country"]})
    return {**state, "structured_data": data}