from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_openai import AzureChatOpenAI
import copy
from pyobjects.pyobj import State


google_search = GoogleSerperAPIWrapper()

model = AzureChatOpenAI(model="gpt-4o-mini", temperature=0)

def fill_missing_fields(state: State) -> State:
    data = copy.deepcopy(state["structured_data"])

    for key, value in data.items():
        if isinstance(value, str) and not value.strip():
            query = f"What is the {key} of {state['company_name']} in {state['country']}?"
            raw_result = google_search.run(query)

            # Use LLM to extract a clean answer from the search result
            prompt = f"""
                            You're an assistant that extracts accurate company information from noisy web content.

                            Search Result Snippet:
                            \"\"\"
                            {raw_result}
                            \"\"\"

                            Extract only the exact value for the field: "{key}".
                            Do not include any additional text or explanations for Ultimate Beneficial Owner, Directors & Shareholders, or Subsidiaries.
                            If the name information of Ultimate Beneficial Owner or Directors & Shareholders, or Subsidiaries is not available, return "Not Available".
                            Return only the extracted value, nothing else. if the information is not available, return "Not Available".
                            Do not include any other information or context.
                            """
            answer = model.invoke(prompt).content.strip()
            data[key] = answer

    return {**state, "structured_data": data}