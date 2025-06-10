# ESG Scores:
# Environmental impact metrics
# Social responsibility indicators
# Governance quality assessment
# Industry-specific ESG benchmarking
from langchain_openai import AzureChatOpenAI
from langchain.tools.tavily_search import TavilySearchResults
from langchain.tools.google_serper import GoogleSerperResults
from pyobjects.pyobj import State
from dotenv import load_dotenv 
load_dotenv()

tavily = TavilySearchResults(max_results=10)
serper = GoogleSerperResults()
llm = AzureChatOpenAI(model='gpt-4o-mini',temperature=0)

def esg_score(state: State) -> State:
    company_name = state.get("company_name")
    country = state.get("country")
    from_date = state.get("from_date")
    to_date = state.get("to_date")

    search_queries = [
        f"ESG scores of {company_name} in {country} from {from_date} to {to_date}",
        f"Environmental impact metrics for {company_name} in {country} from {from_date} to {to_date}",
        f"Social responsibility indicators for {company_name} in {country} from {from_date} to {to_date}",
        f"Environmental, social, and Governance (ESG) quality assessment of {company_name} in {country} from {from_date} to {to_date}",
        f"Industry-specific ESG benchmarking for {company_name} in {country} from {from_date} to {to_date}"
    ]

    # Perform searches
    tavily_results = []
    for query in search_queries:
        results = tavily.run(query)
        if isinstance(results, list):
            tavily_results.extend([r['content'] for r in results if 'content' in r])

    serper_results = []
    for query in search_queries:
        results = serper.run(query)
        if isinstance(results, list):
            serper_results.extend([r['snippet'] for r in results if 'snippet' in r])

    # Combine all search content
    combined_text = "\n".join(tavily_results + serper_results)

    # Refined LLM prompt
    final_prompt = f"""
            You are a Environmental Sustainability analyst tasked with extracting Environment, Social, and Governance (ESG) scores of given company from raw web data.

            Company: {company_name}
            Country: {country}
            Date Range: {from_date} to {to_date}

            Using the context provided, return:
            - Environmental impact metrics
            - Social responsibility indicators
            - Governance quality assessment
            - Industry-specific ESG benchmarking
            
            You can add additional relevant information that may be useful for understanding the ESG scores of the company.
            
            Structure the findings in bullet points or a table for clarity.
            """
    
    response = llm.invoke(final_prompt + "\n\nContext:\n" + combined_text[:12000])  # Truncate if needed

    current_final_data = state.get("final_data", {})
    
    # Add new data
    additional_data = {"esg_score": response.content}
    
    # Merge with existing data
    updated_final_data = {**current_final_data, **additional_data}
    
    return {**state, "final_data": updated_final_data}
    