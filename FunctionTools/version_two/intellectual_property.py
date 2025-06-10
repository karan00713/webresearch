# Intellectual Property Portfolio:
# Patent count and quality
# Trademark portfolio
# IP litigation activity
from langchain_openai import AzureChatOpenAI
from langchain.tools.tavily_search import TavilySearchResults
from langchain.tools.google_serper import GoogleSerperResults
from pyobjects.pyobj import State
from dotenv import load_dotenv 
load_dotenv()

tavily = TavilySearchResults(max_results=10)
serper = GoogleSerperResults()
llm = AzureChatOpenAI(model='gpt-4o-mini',temperature=0)

def intellectual_property(state: State) -> State:
    company_name = state.get("company_name")
    country = state.get("country")
    from_date = state.get("from_date")
    to_date = state.get("to_date")

    search_queries = [
        f"Intellectual property portfolio of {company_name} {country} from {from_date} to {to_date}", 
        f"Patent count and quality of {company_name} {country} from {from_date} to {to_date}",
        f"Trademark portfolio of {company_name} {country}",
        f"IP litigation activity of {company_name} {country} from {from_date} to {to_date}"
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
            You are a Intellectual Property analyst tasked with extracting Intellectual Property details, portfolio and its compliance of given company from raw web data.

            Company: {company_name}
            Country: {country}
            Date Range: {from_date} to {to_date}

            Using the context provided, return:
            - Patent count and quality
            - Trademark portfolio
            - IP litigation activity
            
            You can add additional relevant information that may be useful for understanding the intellectual property portfolio of the company.
            
            Structure the findings in bullet points or a table for clarity.
            """
    
    response = llm.invoke(final_prompt + "\n\nContext:\n" + combined_text[:12000])  # Truncate if needed

    current_final_data = state.get("final_data", {})
    
    # Add new data
    additional_data = {"intellectual_property": response.content}
    
    # Merge with existing data
    updated_final_data = {**current_final_data, **additional_data}
    
    return {**state, "final_data": updated_final_data}
    