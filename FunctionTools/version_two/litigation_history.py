
# Litigation History:
# Active legal proceedings
# Historical cases with outcomes
# Jurisdictional breakdown of legal exposure
from langchain.tools.tavily_search import TavilySearchResults
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_openai import AzureChatOpenAI
from pyobjects.pyobj import State
from datetime import date
from dotenv import load_dotenv
load_dotenv()

tavily = TavilySearchResults(k=10)
serper = GoogleSerperAPIWrapper()
llm = AzureChatOpenAI(model='gpt-4o-mini',temperature=0)

def litigation_history(state: State) -> State:
    company_name = state["company_name"]
    country = state['country'] or ""
    from_date = state.get("from_date", date(2025, 1, 1))
    to_date = state.get("to_date", date.today())
    search_queries = [
        f"Active legal proceedings against {company_name} in {country} between {from_date} and {to_date}",
        f"Historical legal cases involving {company_name} in {country} with outcomes between {from_date} and {to_date}",
        f"Jurisdictional legal exposure of {company_name} in {country} between {from_date} and {to_date}"
    ]

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
            You are an Legal compliance analyst tasked with extracting Litigation History of given company from raw web data.

            Company: {company_name}
            Country: {country}
            Date Range: {from_date} to {to_date}

            Using the context provided, return:
            - Active legal proceedings
            - Historical cases with outcomes
            - Jurisdictional breakdown of legal exposure
            
            You can add additional relevant information that may be useful for understanding the litigation history of the company.

            Structure the findings in bullet points or a table for clarity.
            """
    
    response = llm.invoke(final_prompt + "\n\nContext:\n" + combined_text[:12000])  # Truncate if needed

    current_final_data = state.get("final_data", {})
    
    # Add new data
    additional_data = {"litigation_history": response.content}
    
    # Merge with existing data
    updated_final_data = {**current_final_data, **additional_data}
    
    return {**state, "final_data": updated_final_data}
