from langchain.schema import HumanMessage, SystemMessage
from elsai_core.model import AzureOpenAIConnector
import json
import feedparser
import urllib.parse
import re
import os
import pandas as pd
from newspaper import Article
import requests
import ast
from googlenewsdecoder import new_decoderv1
from pyobjects.pyobj import State
from datetime import date
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_SUBSCRIPTION_KEY = os.getenv("AZURE_SUBSCRIPTION_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

chat = AzureOpenAIConnector().connect_azure_open_ai("gpt-4o-mini")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base="https://api.openai.com/v1",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
)

def generate_search_queries(company, country, data_dir):
    try:
        if country == "":
            prompt = f"""
            For the given company, {company}, use {data_dir["General Details"]} to generate search queries such that it can be used in TAVILY to generate detailed information about the company for adverse media screening and generate corporate actions keywords and adverse media keywords related to this company as separate lists which will be used for classification
            OUTPUT in JSON format:\n"""  + """
            {
                'search_queries': ['query1', 'query2'],
                'corporate_actions':['keyword1', 'keyword2'],
                'adverse_media':['keyword1', 'keyword2']
            }
            """
        else:
            prompt = f"""
            For the given company, {company} in {country}, use {data_dir["General Details"]} to generate search queries such that it can be used in TAVILY to generate detailed information about the company for adverse media screening and generate corporate actions keywords and adverse media keywords related to this company as separate lists which will be used for classification
            OUTPUT in JSON format:\n"""  + """
            {
                'search_queries': ['query1', 'query2'],
                'corporate_actions':['keyword1', 'keyword2'],
                'adverse_media':['keyword1', 'keyword2']
            }
            """
        messages = [
                    SystemMessage(content = "You are a search query and keyword generator for adverse media screening."),
                    HumanMessage(content = prompt)
                ]
        
        try:    
            response = chat(messages)
            result = response.content
            result = result.replace("```json", "").replace("```", "")
            queries = json.loads(result)
            print("Generated search queries successfully.")
            return queries
        except Exception as e:
            print(f"Error parsing JSON response: {str(e)}")
            return None

    except Exception as e:
        print(f"Error generating search queries: {str(e)}")
        return None
    
def find_tag(content, corporate_actions, adverse_media= []):
    query = f"""Find the tag from the following list related to the given company in the provided content. If not found, return an empty list.
    Tags: {corporate_actions}, {adverse_media}
    Content: {content}""" + """
    
    OUTPUT IN JSON format:
    {
        "tags"= [tag1, tag2]
    }
    
    or 
    {   
        "tags"= []
    }
    """
    messages = [
                SystemMessage(content = "You are a tag finder."),
                HumanMessage(content = query)
            ]
    
    try:          
        response = chat(messages)
        result = response.content
        result = result.replace("```json", "").replace("```", "")
        tags = json.loads(result)
        return tags["tags"]
    except Exception as e:
        print(f"Error parsing JSON response: {str(e)}")
        return None
    
def news_articles(search_queries, df, company, corporate_actions, adverse_media, fromDate, toDate, max_results):
    def fetch_news_urls(query, fromDate, toDate, num_results=max_results):
        """
        Fetch news article URLs from a given search query using Google News RSS feed
        with a date filter for the last 5 years
        
        Args:
            query (str): The search query
            num_results (int): Maximum number of results to return
            years_back (int): How many years back to search for articles
            
        Returns:
            list: List of article URLs and their publication dates
        """
        end_date = toDate
        start_date = fromDate
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        date_query = f"{query} after:{start_date_str} before:{end_date_str}"
        encoded_query = urllib.parse.quote(date_query)
        
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(rss_url)
            results = []
            for entry in feed.entries[:num_results]:

                if 'link' in entry:
                    match = re.search(r'url=([^&]+)', entry.link)
                    if match:
                        actual_url = urllib.parse.unquote(match.group(1))
                        url = actual_url
                    else:
                        url = entry.link
                                            
                    pub_date = entry.get('published', 'Date unknown')                    
                    results.append({
                        'url': url,
                        'title': entry.get('title', 'No title'),
                        'published': pub_date
                    })
            return results
        
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def get_content(url):
        try:
            decoded_url = new_decoderv1(url)
            if decoded_url.get("status"):
                link=decoded_url["decoded_url"]
                article = Article(link)
                article.download()
                article.parse()
                return([link, article.text])
            else:
                return ""
        except Exception as e:
            print(f"Cannot fetch content for {url}: {e}")
            return ""
        
    def get_content_summary(text):
        try:
            prompt = f"""
            Summarize the given content in a few sentences. Every key point must be included in the summary. Return only the text.
            Content: {text}
            
            OUTPUT FORMAT:
            Summary
            """
            messages = [
                        SystemMessage(content = "You are a summarisation AI."),
                        HumanMessage(content = prompt)
                    ]                    
            summary = chat(messages)
            return summary.content
        
        except Exception as e:
            print(f"Error summarizing content: {e}")
            return ""
        
    def get_sentiment(text, company):
        prompt = f"Analyze the sentiment of the given content related to {company} and classify it as 'Positive', 'Negative', or 'Neutral'. Classify it as negative only if the content reflects negatively on the company. Provide only one of these three labels as the response, without any additional text or explanations. Content: {text}"
        messages = [
                        SystemMessage(content = "You are a sentiment analysis AI."),
                        HumanMessage(content = prompt)
                    ]                    
        sentiment = chat(messages)
        return sentiment.content
    
    def check_content(content, company):
        prompt = f"""
        Analyze the content of the given text and check if the text is related to the company '{company}' or not. If it is not related, return empty string, else return the content as such. 
        Content: {content}
        
        OUTPUT FORMAT:
        {content} if it is related to '{company}', 
        "", otherwise
        """
        messages = [
                    SystemMessage(content = "You are a content analysis AI."),
                    HumanMessage(content = prompt)
                ]
        response = chat(messages)
        return response.content
    
    def news(search_query, company, df_news, visited_urls):
        try:
            article_urls = fetch_news_urls(search_query, fromDate, toDate)
        
            print(f"\nFound {len(article_urls)} news articles for '{search_query}':")
            i= 0
            for url in article_urls:
                result=get_content(url["url"])
                link=result[0]
                content=result[1]
                if content == "" or content == None:
                    continue
                if link in visited_urls:
                    continue
                visited_urls.add(link)
                summary=get_content_summary(content)
                summarised_content = check_content(summary, company)
                if summarised_content == '""':
                    continue
                sentiment= get_sentiment(summarised_content, company)
                if sentiment == "Positive":
                    tag = find_tag(content, corporate_actions, [])
                else:
                    tag = find_tag(content, corporate_actions, adverse_media)
                df_news.loc[i]= [link, summary, sentiment, tag]
                i+= 1
            return [visited_urls, df_news]
        except Exception as e:
            print(f"Error fetching news: {e}")
            return None
    
    visited_urls = set()
    for query in search_queries:
        try:
            df_news = pd.DataFrame(columns=["url", "content", "sentiment", "tags"])
            result = news(query, company, df_news, visited_urls)
            if result is None:
                continue
            visited_urls = result[0]
            df_news = result[1]
            df = pd.concat([df, df_news], ignore_index=True)
        except Exception as e:
            print(f"Error processing news: {e}")
    return df

def articles(company, corporate_actions, adverse_media, max_results, fromDate, toDate):
    entities = [company]
    adverse_keywords = adverse_media
    non_adverse_keywords = ["award", "recognition", "innovation", "sustainability", "CSR", "growth"]
    corporate_actions = corporate_actions
    
    def sentiment_analysis(final_analysis, company):
        if not final_analysis.strip():
            return "No recent mentions found."
    
        prompt = f"""
        Analyze the sentiment of the given content for {company} and classify it as 'Positive', 'Negative', or 'Neutral'. 
        Classify it as negative only if the content reflects negatively on the company.
    
        **Rules:**
        - Classify sentiment as **Positive, Negative, or Neutral**.
        - Provide a brief explanation of why this sentiment was assigned.
    
        **Text:**  
        {final_analysis}
    
        **Output JSON Format:**  
        {{
            "sentiment": "Positive" or "Negative" or "Neutral",
            "explanation": "Brief reason for classification"
        }}
        """
    
        messages = [
            SystemMessage(content="You are a sentiment analysis AI."),
            HumanMessage(content=prompt)
        ]
    
        response = chat(messages)
        return response.content
    
    def search_tavily_adverse(entity, toDate, fromDate):
        try:
            results = []
            if fromDate and toDate:
                date_filter = f" after:{fromDate} before:{toDate}"
            elif fromDate:
                date_filter = f" after:{fromDate}"
            elif toDate:
                date_filter = f" before:{toDate}"
            for keyword in adverse_keywords+non_adverse_keywords+corporate_actions:
                query = f"{entity} {keyword} {date_filter}"
                url = "https://api.tavily.com/search"
                payload = {
                    "api_key": os.getenv("TAVILY_API_KEY"),
                    "query": query,
                    "max_results": max_results
                }
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    results.extend(response.json().get("results", []))
                else:
                    print(f"Error searching Tavily for {query}: {response.status_code}")
            return results
        except Exception as e:
            print(f"Error searching Tavily: {e}")
            return []

    def analyze_with_gpt(text):
        try:
            prompt = f"""
            Summarize the given content in a few sentences. Every key point must be included in the summary. Return only the text.
            Content: {text}
            
            OUTPUT FORMAT:
            Summary
            """
            messages = [
                        SystemMessage(content = "You are a summarisation AI."),
                        HumanMessage(content = prompt)
                    ]                    
            summary = chat(messages)
            return summary.content
        
        except Exception as e:
            print(f"Error summarizing content: {e}")
            return None

    def adverse_media_screening(entities, company, fromDate, toDate):
        results = []
    
        for entity in entities:
            print(f"Searching for adverse media related to {entity}...")
            search_results = search_tavily_adverse(f"{entity} news", fromDate, toDate)
            visited_urls = set()
            for result in search_results:
                if result.get("url", "") in visited_urls:
                    continue
                visited_urls.add(result.get("url", ""))
                content = result.get("content", "")
                if content:
                    analysis = analyze_with_gpt(content)
                    if analysis == '""':
                        continue
                    sentiment = sentiment_analysis(analysis, company)
                    json_sentiment = json.loads(sentiment)
                    print(json_sentiment)
                    if json_sentiment["sentiment"] == "Positive":
                        tag = find_tag(content, corporate_actions, [])
                    else:
                        tag = find_tag(content, corporate_actions, adverse_media)
                    results.append({
                        "url": result.get("url", ""),
                        "content": analysis,
                        "sentiment": json_sentiment["sentiment"],
                        "tags": tag
                    })
    
        return pd.DataFrame(results)
    
    df = adverse_media_screening(entities, company, fromDate, toDate)
    return df

def get_analysis_results(content_list, company):
    prompt = f"""
    Analyse the following content and identify the key findings related to company, {company}, from the list provided. Return maximum 15 key findings as bullet points. Make sure that the key findings are unique and related to {company}. Do not include any other text other than the key findings.
    Content: {content_list}
    
    OUTPUT FORMAT:
    "- Key Finding 1\n
    - Key Finding 2"
    if key findings are found
    
    ""
    otherwise
    """
    response = llm.invoke(prompt)
    return response.content

def director_check(content, company, data_dict):
    prompt = f"""
                From {data_dict}, for {company}, identify the directors. From the information, perform director sanity check on the provided content below. 
                Return every content that refers to the directors of the company. From that content, analyse it and provide bullet points related to the directors only.
                Do not include any other text other than the director check analysis. Return as bullet points for markdown file.
                Content: {content}
                OUTPUT FORMAT:
                - Point 1
                - Point 2
                """
    response = llm.invoke(prompt)
    return response.content

def analyze_sentiment_by_tag(df):
    def parse_tags(tag_str):
        try:
            if isinstance(tag_str, str):
                return ast.literal_eval(tag_str)
            return tag_str
        except (ValueError, SyntaxError):
            return []
    
    processed_df = df.copy()
    processed_df['parsed_tags'] = processed_df['tags'].apply(parse_tags)
    
    tag_counts = {}
    
    for _, row in processed_df.iterrows():
        tags = row['parsed_tags']
        sentiment = row['sentiment']
        
        if not isinstance(tags, list) or len(tags) == 0:
            continue
        
        for tag in tags:
            tag = tag.capitalize()
            if tag not in tag_counts:
                tag_counts[tag] = {
                    'total': 0,
                    'Positive': 0,
                    'Negative': 0,
                    'Neutral': 0
                }
            
            tag_counts[tag]['total'] += 1
            tag_counts[tag][sentiment] += 1
    
    result_rows = []
    for tag, counts in tag_counts.items():
        total = counts['total']
        row_data = {'tag': tag, 'total_articles': total}
        
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            count = counts.get(sentiment, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            row_data[sentiment] = round(percentage, 2)
        
        result_rows.append(row_data)
    
    if not result_rows:
        return pd.DataFrame(columns=['Negative', 'Neutral', 'Positive', 'total_articles'])
    
    result_df = pd.DataFrame(result_rows)
    result_df = result_df.set_index('tag')
    result_df = result_df[['Negative', 'Neutral', 'Positive', 'total_articles']]
    result_df = result_df.drop(columns=['total_articles'])
    
    return result_df

def process_additional_research(state: State):
        # Generate search queries
    company_name = state["company_name"]
    country = state['country'] or ""
    from_date = state.get("from_date", date(2025, 1, 1))
    to_date = state.get("to_date", date.today())
    queries = generate_search_queries(company_name, country, state["structured_data"])
    max_results = 2  # Set maximum results for each query
    # Initialize DataFrame
    df = pd.DataFrame(columns=["url", "content", "sentiment", "tags"])
    
    # Perform web research
    corporate_actions = queries["corporate_actions"]
    adverse_media = queries["adverse_media"]
    df = news_articles(queries["search_queries"], df, company_name, corporate_actions, adverse_media, from_date, to_date, max_results)
    df_articles = articles(company_name, corporate_actions, adverse_media, max_results, from_date, to_date)
    df = pd.concat([df, df_articles], ignore_index=True)
    
    # Sentiment analysis
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    
    # Analyze content by sentiment
    positive_content = df[df['sentiment'] == 'Positive']['content'].tolist()
    negative_content = df[df['sentiment'] == 'Negative']['content'].tolist()
    neutral_content = df[df['sentiment'] == 'Neutral']['content'].tolist()
    
    positive_content = get_analysis_results(positive_content, company_name)
    negative_content = get_analysis_results(negative_content, company_name)
    neutral_content = get_analysis_results(neutral_content, company_name)
    
    # Director check
    content_list = df["content"].tolist()
    director_content = director_check(content_list, company_name, state["structured_data"])
    
    # Sentiment by tag
    sent_df = analyze_sentiment_by_tag(df)
    
    # Prepare markdown content
    markdown_content = f"# Adverse Media Research Results\n\n"
    markdown_content += sentiment_counts.to_markdown(index=False)
    
    if positive_content != '""':
        markdown_content += "\n\n## Positive Media Keypoints:\n"
        markdown_content += positive_content
    
    if negative_content != '""':
        markdown_content += "\n\n## Negative Media Keypoints:\n"
        markdown_content += negative_content
    
    if neutral_content != '""':
        markdown_content += "\n\n## Neutral Media Keypoints:\n"
        markdown_content += neutral_content
    
    markdown_content += "\n\n## Sentiment Distribution by Category\n"
    markdown_content += sent_df.to_markdown()
    
    markdown_content += "\n\n## Directors Sanity Check\n"
    markdown_content += director_content

    # Final response data
    final_response_data = {
        "web_research_results": df.to_dict(orient="records"),
        "sentiment_analysis": {
            "sentiment_counts": sentiment_counts.to_dict(orient="records"),
            "sentiment_by_tag": sent_df.to_dict()
        },
        "markdown_content": markdown_content,
        "status": "research_complete"
    }
    return {**state, "final_data": final_response_data}