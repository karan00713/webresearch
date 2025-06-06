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
            structured_data = result.get("structured_data", {})

            st.success("✅ Extraction complete!")
            st.subheader("📋 Structured Company Data")
            st.write(f"🕒 Time taken: {elapsed_time:.2f} seconds")
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

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            traceback.print_exc()
else:
    st.info("Please enter the required details and click submit.")
