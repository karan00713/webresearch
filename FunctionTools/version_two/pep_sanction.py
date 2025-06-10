from langchain_openai import AzureChatOpenAI
from langchain.tools.tavily_search import TavilySearchResults
from langchain.tools.google_serper import GoogleSerperResults
from pyobjects.pyobj import State
from dotenv import load_dotenv 
load_dotenv()

tavily = TavilySearchResults(max_results=10)
serper = GoogleSerperResults()
llm = AzureChatOpenAI(model='gpt-4o-mini',temperature=0)

def pep_and_sanction(state: State) -> State:
    company_name = state.get("company_name")
    country = state.get("country")
    from_date = state.get("from_date")
    to_date = state.get("to_date")

    search_queries = [
        f"{company_name} {country} politically exposed persons executives beneficial owners",
        f"{company_name} {country} sanction list OFAC EU UN {from_date} to {to_date}",
        f"{company_name} PEP exposure leadership risk analysis",
        f"{company_name} {country} global sanctions news"
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
            You are a PEP and Sanction List analyst tasked with extracting PEP and Sanction-related risk intelligence from raw web data.

            Company: {company_name}
            Country: {country}
            Date Range: {from_date} to {to_date}

            Using the context provided, return:
            - PEP Checks: List of known politically exposed persons linked to the company, their roles, jurisdictions, and risk indications.
            - Sanction Checks: Presence in any sanction lists (OFAC, UN, EU, etc.), jurisdictions, verification dates, and any associated individuals/entities.
            
            You can add additional relevant information that may be useful for understanding the litigation history of the company.
            
            Structure the findings in bullet points or a table for clarity.
            """
    
    response = llm.invoke(final_prompt + "\n\nContext:\n" + combined_text[:12000])  # Truncate if needed
    
    current_final_data = state.get("final_data", {})
    
    # Add new data
    additional_data = {"pep_sanction_analysis": response.content}
    
    # Merge with existing data
    updated_final_data = {**current_final_data, **additional_data}
    
    return {**state, "final_data": updated_final_data}
    