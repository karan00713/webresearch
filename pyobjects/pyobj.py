from typing import TypedDict, Any, Optional
from datetime import date

class State(TypedDict):
    company_name: str
    country: str
    web_content: str 
    structured_data: dict 
    source_list: Any
    from_date: Optional[date]
    to_date: Optional[date]
    final_data: Optional[dict]