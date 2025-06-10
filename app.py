import streamlit as st
import pandas as pd
from main import app
import traceback
import time

st.set_page_config(page_title="Web Research Media Agent", layout="centered")

st.title("🔍 Web Research Media Agent")
st.markdown("Enter a company name and country to fetch structured data from the web.")

# Input form
with st.form("company_form"):
    company_name = st.text_input("Company Name", placeholder="e.g., Infosys Limited")
    country = st.text_input("Country", placeholder="e.g., India")
    from_date = st.date_input("From Date", value=None, help="Optional: Start date for the search")
    to_date = st.date_input("To Date", value=None, help="Optional: End date for the search")
    submitted = st.form_submit_button("Search & Extract")

# Process input
if submitted and company_name and country:
    with st.spinner("Searching and extracting company information..."):
        state = {"company_name": company_name, "country": country,
                 "from_date": from_date, "to_date": to_date}
        
        try:
            start_time = time.time()
            result = app.invoke(state)
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_minutes = elapsed_time / 60
            structured_data = result.get("structured_data", {})

            st.success("✅ Extraction complete!")
            st.subheader("📋 Structured Company Data")
            st.write(f"🕒 Time taken: {elapsed_minutes:.2f} minutes")
            for key, value in structured_data.items():
                st.markdown(f"**{key}**: {value if value else 'N/A'}")
            if "source_list" in result and result["source_list"]:
                st.markdown("### 📚 Sources")
                for url in result["source_list"]:
                    st.markdown(f"- [🔗 {url}]({url})")
            final_data = result.get("final_data", {})
            # Display research results
            st.header("Web Research Results")
            df_research = pd.DataFrame(final_data["web_research_results"])
            st.dataframe(df_research)

            # Display sentiment counts
            st.header("Sentiment Counts")
            df_counts = pd.DataFrame(final_data["sentiment_analysis"]["sentiment_counts"])
            st.dataframe(df_counts)

            # Display sentiment by tag
            st.header("Sentiment by Tag")
            df_by_tag = pd.DataFrame(final_data["sentiment_analysis"]["sentiment_by_tag"])
            st.dataframe(df_by_tag)

            # Display markdown content
            st.header("Summary")
            st.markdown(final_data["markdown_content"])
            
            # PEP and Sanction Data
            st.header("PEP and Sanction")
            st.write(final_data["pep_sanction_analysis"])
            
            # Litigation History
            st.header("Litigation History")
            st.write(final_data["litigation_history"]) 
            
            # Regulatory Compliance
            st.header("Regulatory Compliance")
            st.write(final_data["regulatory_compliance"])
            
            # Major Customers
            st.header("Major Customers")
            st.write(final_data["major_customers"])
            
            # Suppliers and Vendors
            st.header("Suppliers and Vendors")
            st.write(final_data["suppliers_vendors"])
            
            # Market Share
            st.header("Market Share")
            st.write(final_data["market_share"])
            
            # ESG Score
            st.header("ESG Score")
            st.write(final_data["esg_score"])
            
            # Intellectual Property
            st.header("Intellectual Property")
            st.write(final_data["intellectual_property"])
            
            # Management Changes
            st.header("Management Changes")
            st.write(final_data["management_changes"])
            
            # Geographic Risk Assessment
            st.header("Geographic Risk Assessment")
            st.write(final_data["geographic_risk_assessment"])
            
            # Industry Risk Assessment
            st.header("Industry Risk Assessment")
            st.write(final_data["industry_risk_assessment"])
            
            # Competitors Analysis
            st.header("Competitors Analysis")
            st.write(final_data["competitors_analysis"])

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            traceback.print_exc()
else:
    st.info("Please enter the required details and click submit.")
