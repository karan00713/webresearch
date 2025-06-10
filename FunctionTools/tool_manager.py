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
class ToolManager:
    """
    ToolManager is responsible for managing and loading function tools.
    It provides a method to load a specific function tool based on its name.
    """
    @staticmethod
    def get(function_name:str):
        try:
            tools = {
                "BASIC_DETAILS": basic_details,
                "EXTRACT_DATA": extract_data,
                "FILL_MISSING": fill_missing_fields,
                "ADVERSE_MEDIA": adverse_media_analysis,
                "PEP_SANCTION": pep_and_sanction,               
                "LITIGATION_HISTORY": litigation_history,
                "REGULATORY_COMPLIANCE": regulatory_compliance,
                "MAJOR_CUSTOMERS": major_customers,
                "SUPPLIERS_VENDORS": suppliers_and_vendors,
                "MARKET_SHARE": market_share,
                "ESG_SCORE": esg_score,
                "INTELLECTUAL_PROPERTY": intellectual_property,
                "MANAGEMENT_CHANGES": management_changes,
                "GEOGRAPHIC_RISK": geographic_risk_assessment,               
                "INDUSTRY_RISK": industry_risk_assessment,
                "COMPETITORS_ANALYSIS": competitors_analysis
            
            }
            return tools[function_name]
        except KeyError:
            raise ValueError(f"Unknown function tool: {function_name}")