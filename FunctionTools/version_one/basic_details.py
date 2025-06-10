from langchain.tools.tavily_search import TavilySearchResults
from pyobjects.pyobj import State

tavily_tool = TavilySearchResults(k=10)
def basic_details(state: State) -> State:
    try:
        query = f"Company details of {state['company_name']} in {state['country']}"
        tavily_result = tavily_tool.invoke({"query": query})
        content = "\n\n".join([r["content"] for r in tavily_result])
        urls = [r["url"] for r in tavily_result if "url" in r]
        return {**state, "web_content": content, "source_list": urls}
    except Exception as e:
        content = f"Search failed: {e}"
        return {**state, "web_content": content }