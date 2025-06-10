from langgraph.graph import StateGraph, START,END
from langchain_core.runnables import RunnableLambda
from pyobjects.pyobj import State
from FunctionTools.version_one.basic_details import basic_details
from FunctionTools.version_one.extract_data import extract_data
from FunctionTools.version_one.fill_missing import fill_missing_fields
from FunctionTools.version_one.adverse_media import adverse_media_analysis
from FunctionTools.version_two.pep_sanction import pep_and_sanction
from FunctionTools.version_two.litigation_history import litigation_history
from FunctionTools.version_two.regulatory_compliance import regulatory_compliance
from FunctionTools.version_two.major_customers import major_customers
from FunctionTools.version_two.suppliers_vendors import suppliers_and_vendors
from FunctionTools.version_two.market_share import market_share
from FunctionTools.version_two.esg_score import esg_score
from FunctionTools.version_two.intellectual_property import intellectual_property
from FunctionTools.version_two.management_changes import management_changes
from FunctionTools.version_two.geographic_risk import geographic_risk_assessment
from FunctionTools.version_two.industry_risk import industry_risk_assessment
from FunctionTools.version_two.competitors_analysis import competitors_analysis
from dotenv import load_dotenv

load_dotenv()

# LangGraph node setup
graph = StateGraph(State)
graph.add_node("search", RunnableLambda(basic_details))
graph.add_node("extract", RunnableLambda(extract_data))
graph.add_node("fill_missing", RunnableLambda(fill_missing_fields))
graph.add_node("adverse_media", RunnableLambda(adverse_media_analysis))
graph.add_node("pep_sanction", RunnableLambda(pep_and_sanction))
graph.add_node("litigation_history", RunnableLambda(litigation_history))
graph.add_node("regulatory_compliance", RunnableLambda(regulatory_compliance))
graph.add_node("major_customers", RunnableLambda(major_customers))
graph.add_node("suppliers_vendors", RunnableLambda(suppliers_and_vendors))  
graph.add_node("market_share", RunnableLambda(market_share))
graph.add_node("esg_score", RunnableLambda(esg_score))
graph.add_node("intellectual_property", RunnableLambda(intellectual_property))
graph.add_node("management_changes", RunnableLambda(management_changes))
graph.add_node("geographic_risk_assessment", RunnableLambda(geographic_risk_assessment))
graph.add_node("industry_risk_assessment", RunnableLambda(industry_risk_assessment))
graph.add_node("competitors_analysis", RunnableLambda(competitors_analysis))

# Define the state graph structure
graph.add_edge(START, "search")
graph.add_edge("search", "extract")
graph.add_edge("extract", "fill_missing")
graph.add_edge("fill_missing", "adverse_media")
graph.add_edge("adverse_media", "pep_sanction")
graph.add_edge("pep_sanction", "litigation_history")
graph.add_edge("litigation_history", "regulatory_compliance")
graph.add_edge("regulatory_compliance", "major_customers")
graph.add_edge("major_customers", "suppliers_vendors")
graph.add_edge("suppliers_vendors", "market_share")
graph.add_edge("market_share", "esg_score")
graph.add_edge("esg_score", "intellectual_property")
graph.add_edge("intellectual_property", "management_changes")
graph.add_edge("management_changes", "geographic_risk_assessment")
graph.add_edge("geographic_risk_assessment", "industry_risk_assessment")
graph.add_edge("industry_risk_assessment", "competitors_analysis")
graph.add_edge("competitors_analysis", END)

app = graph.compile()