import streamlit as st
import pandas as pd
from pandas.core.frame import DataFrame
from youtube_comment_downloader import *
import io
import openai

# Function to summarize text using OpenAI ChatGPT API
def summarize_text(input_text: str, api_key: str, max_tokens: int = 100) -> str:
    """
    Summarizes the given text using OpenAI's GPT API.
    
    Args:
        input_text (str): The text to summarize.
        api_key (str): Your OpenAI API key.
        max_tokens (int): Maximum tokens for the summary.
    
    Returns:
        str: The summarized text.
    """
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Summarize this text: {input_text}"}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error during summarization: {e}"

@st.cache_data
def youtube_url_to_df(Youtube_URL: str) -> DataFrame:
    """Fetch comments from a YouTube video and return as a DataFrame."""
    try:
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(Youtube_URL, sort_by=SORT_BY_POPULAR)
        
        all_comments_dict = {
            'cid': [],
            'text': [],
            'time': [],
            'author': [],
            'channel': [],
            'votes': [],
            'replies': [],
            'photo': [],
            'heart': [],
            'reply': [],
            'time_parsed': []
        }
        
        for comment in comments:
            for key in all_comments_dict.keys():
                all_comments_dict[key].append(comment[key])
        
        comments_df = pd.DataFrame(all_comments_dict)
        return comments_df
    
    except Exception as error:
        if Youtube_URL != "":
            st.exception(error)
        return None

def download_df(df: DataFrame, label: str) -> None:
    """Add a button to download the DataFrame."""
    format_download = st.radio("Choose download format:", ['CSV', 'Excel'])
    
    if format_download == 'CSV':
        download_format = 'text/csv'
        file_extension = 'csv'
        data_df = df.to_csv(index=False)
    elif format_download == 'Excel':
        download_format = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        file_extension = 'xlsx'
        excel_bytes = io.BytesIO()
        data_df = excel_bytes
        
    try:
        st.download_button(label=f"Download {label} DataFrame ({format_download})", 
                           data=data_df, 
                           file_name=f'dataframe.{file_extension}', 
                           mime=download_format)
    except Exception as error:
        st.exception(error)

def main():
    st.header("YouTube Comments Downloader and Summarizer")
    st.write("Download and summarize comments from YouTube videos effortlessly.")
    
    st.divider()
    
    url_text = st.text_input("Enter YouTube URL")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    raw_df = youtube_url_to_df(url_text)
    
    if raw_df is None or raw_df.empty:
        st.info('Please enter a valid YouTube link.')
    else:
        download_df(raw_df, "Raw")
        
        # Display raw comments
        st.subheader("Raw Comments")
        st.dataframe(raw_df[['text']].head(10))  # Show top 10 comments
        
        # Combine all comments into one text for summarization
        combined_comments = " ".join(raw_df['text'].tolist())
        
        if api_key:
            with st.spinner("Summarizing comments..."):
                summary = summarize_text(combined_comments, api_key)
                st.subheader("Summary of Comments")
                st.write(summary)
        else:
            st.warning("Please provide your OpenAI API key to generate summaries.")

main()
