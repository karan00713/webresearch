from langgraph.graph import StateGraph, START,END
from langchain_core.runnables import RunnableLambda
from FunctionTools.additionalresearch import process_additional_research
from FunctionTools.searchtavily import search_with_tavily
from FunctionTools.extractdata import extract_data
from FunctionTools.fillmissing import fill_missing_fields
from pyobjects.pyobj import State
from dotenv import load_dotenv

load_dotenv()

# LangGraph node setup
graph = StateGraph(State)
graph.add_node("search", RunnableLambda(search_with_tavily))
graph.add_node("extract", RunnableLambda(extract_data))
graph.add_node("fill_missing", RunnableLambda(fill_missing_fields))
graph.add_node("additional_research", RunnableLambda(process_additional_research))

# Define the state graph structure
graph.add_edge(START, "search")
graph.add_edge("search", "extract")
graph.add_edge("extract", "fill_missing")
graph.add_edge("fill_missing", "additional_research")
graph.add_edge("additional_research", END)

app = graph.compile()