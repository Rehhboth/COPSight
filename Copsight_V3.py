import gspread
from google.oauth2 import service_account
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import streamlit as st
import os
import csv
import time
import logging
import requests
from typing import Dict, Any, Tuple
import io
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from fpdf import FPDF
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add at the beginning of the file, after imports
if 'evaluation_running' not in st.session_state:
    st.session_state.evaluation_running = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
SHEET_ID = os.getenv('SHEET_ID')
OUTPUT_SHEET_ID = os.getenv('OUTPUT_SHEET_ID')
PROMPTS_SHEET_ID = os.getenv('PROMPTS_SHEET_ID')
EXPECTED_HEADERS = ['Ticket ID', 'CSA', 'conversation_text']
API_KEY = os.getenv('OPENAI_API_KEY')
API_URL = 'https://api.openai.com/v1/chat/completions'
MAX_WORKERS = 5
RATE_LIMIT = 60
BATCH_SIZE = 10

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Add Slack configuration
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_API_URL = "https://slack.com/api"

# Function to format time in a human-readable way
def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} seconds"
    minutes = math.floor(seconds / 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes} minute{'s' if minutes != 1 else ''} {remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"

def show_loading_indicator(container=None, message="Loading..."):
    """
    Shows a standardized loading indicator with GIF and message.
    Returns the container for later cleanup.
    """
    if container is None:
        container = st.empty()
    
    with container:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("https://i.ibb.co/zn1gJHQ/loading.gif", width=50)
        with col2:
            st.markdown(f"<p style='margin-top: 15px;'>{message}</p>", unsafe_allow_html=True)
    
    return container

def hide_loading_indicator(container):
    """Hides the loading indicator by clearing the container."""
    if container:
        container.empty()

# Function to fetch and format conversations from Google Sheet
def fetch_conversations():
    loading_container = None
    try:
        loading_container = show_loading_indicator(message="Fetching conversations from Google Sheet...")
        
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_JSON,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(SHEET_ID).worksheet('Raw Data')
        data = sheet.get_all_records()
        df = pd.DataFrame(data)

        # Print available columns for debugging
        logging.info(f"Available columns in raw data: {df.columns.tolist()}")

        # Check if required columns exist
        required_columns = ['Ticket ID', 'CSA', 'conversation_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}. Available columns: {df.columns.tolist()}"
            logging.error(error_msg)
            hide_loading_indicator(loading_container)
            raise ValueError(error_msg)

        # Only keep the columns needed for evaluation
        output_df = df[required_columns]

        # Ensure data types are correct
        output_df['Ticket ID'] = output_df['Ticket ID'].astype(str)
        output_df['CSA'] = output_df['CSA'].astype(str)
        output_df['conversation_text'] = output_df['conversation_text'].astype(str)

        unique_conversation_ids = output_df['Ticket ID'].nunique()
        hide_loading_indicator(loading_container)
        return output_df, unique_conversation_ids
    except Exception as e:
        if loading_container:
            hide_loading_indicator(loading_container)
        logging.error(f"Error in fetch_conversations: {e}")
        raise e 

# Function to load prompts from Google Sheet
def load_prompts():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_JSON,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(PROMPTS_SHEET_ID).worksheet('Prompts')
    headers = sheet.row_values(1)  # Headers in row 1 (A1:E1)
    prompts_row = sheet.row_values(2)  # Prompts in row 2 (A2:E2)
    if len(headers) != len(prompts_row):
        raise ValueError("Mismatch between headers and prompts.")
    prompts = dict(zip(headers, prompts_row))
    return prompts

# Function to evaluate with GPT
def evaluate_with_gpt(full_prompt: str, conversation: str) -> Tuple[int, str]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    data = {
        'model': 'gpt-4o-mini',
        'messages': [
            {'role': 'system', 'content': "You are an AI assistant tasked with evaluating customer service conversations. You must provide a whole number rating between 0 and 5 (inclusive). No decimal points or fractions are allowed."},
            {'role': 'user', 'content': full_prompt + "\n\nConversation:\n" + conversation}
        ]
    }
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()
        completion = response.json()['choices'][0]['message']['content'].strip()
        logging.info(f"GPT response: {completion}")

        # Extract the first Rating and its associated Reasoning (non-greedy)
        rating_match = re.search(r'Rating:\s*(\d+)', completion)
        reasoning_match = re.search(
           r'Reasoning:\s*([\s\S]*?)(?=\nRating:|$)',  # non-greedy up to next "Rating:" or end of text
            completion
        )

        if rating_match and reasoning_match:
            rating = int(rating_match.group(1))
            # Ensure rating is between 0 and 5
            rating = max(0, min(5, rating))
            reasoning = reasoning_match.group(1).strip()
            return rating, reasoning
        else:
            logging.error("Invalid response format")
            return 0, "Invalid response"

    except requests.exceptions.RequestException as e:
        logging.error(f"Request exception occurred: {e}")
        st.error(f"Request exception occurred: {e}")
        return 0, "Request error"
    except (KeyError, IndexError) as e:
        logging.error(f"Error parsing GPT response: {e}")
        st.error(f"Error parsing GPT response: {e}")
        return 0, "Parsing error"

# Function to analyze grammar using TextBlob and NLTK
def analyze_grammar(text: str) -> Tuple[float, str]:
    """
    Analyze grammar using TextBlob and NLTK.
    Returns a tuple of (score, feedback).
    """
    try:
        blob = TextBlob(text)
        sentences = sent_tokenize(text)
        
        # Grammar score based on TextBlob's spell checking
        grammar_score = 0
        feedback_points = []
        
        # Check spelling
        for word in blob.words:
            if not word.isalpha():
                continue
            if word.lower() not in stopwords.words('english'):
                if not blob.words.spellcheck()[0][1] > 0.9:
                    feedback_points.append(f"Possible spelling error: {word}")
        
        # Check sentence structure
        for sentence in sentences:
            if len(sentence.split()) < 3:
                feedback_points.append("Very short sentence detected")
            if len(sentence.split()) > 30:
                feedback_points.append("Very long sentence detected")
        
        # Calculate score (5 is perfect, 0 is worst)
        grammar_score = max(0, 5 - len(feedback_points) * 0.5)
        
        return grammar_score, "\n".join(feedback_points) if feedback_points else "Good grammar and sentence structure"
    except Exception as e:
        logging.error(f"Error in grammar analysis: {e}")
        return 0, "Error analyzing grammar"

# Function to analyze tone using VADER and TextBlob
def analyze_tone(text: str) -> Tuple[float, str]:
    """
    Analyze tone using VADER and TextBlob.
    Returns a tuple of (score, feedback).
    """
    try:
        # Get VADER sentiment scores
        vader_scores = vader_analyzer.polarity_scores(text)
        
        # Get TextBlob sentiment
        blob = TextBlob(text)
        
        # Combine scores
        tone_score = (vader_scores['compound'] + 1) * 2.5  # Convert to 0-5 scale
        
        # Generate feedback
        feedback = []
        if vader_scores['neg'] > 0.1:
            feedback.append("Some negative language detected")
        if vader_scores['pos'] > 0.1:
            feedback.append("Positive language detected")
        if vader_scores['neu'] > 0.8:
            feedback.append("Very neutral tone")
        
        return tone_score, "\n".join(feedback) if feedback else "Appropriate tone for customer service"
    except Exception as e:
        logging.error(f"Error in tone analysis: {e}")
        return 0, "Error analyzing tone"

# Function to evaluate conversation and return evaluation data
def evaluate_conversation(row: pd.Series, prompts: Dict[str, str]) -> Dict[str, Any]:
    conversation = row['conversation_text']
    csa_name = row['CSA']
    ticket_id = row['Ticket ID']
    evaluation_results = {}
    
    # Pre-analyze grammar and tone
    grammar_score, grammar_feedback = analyze_grammar(conversation)
    tone_score, tone_feedback = analyze_tone(conversation)
    
    for criteria, criteria_prompt in prompts.items():
        if criteria == "Grammar & Punctuation":
            evaluation_results[f"{criteria} Rating"] = grammar_score
            evaluation_results[f"{criteria} Reasoning"] = grammar_feedback
            continue
        elif criteria == "Tone of voice":
            evaluation_results[f"{criteria} Rating"] = tone_score
            evaluation_results[f"{criteria} Reasoning"] = tone_feedback
            continue
            
        full_prompt = f"""
Rate the response of CSA from Laundryheap based on the following criteria. Provide a rating from 0 to 5 and a brief reasoning for the rating.

Criteria: {criteria_prompt}

Ensure the output is in the format:

Rating: [number]
Reasoning: [brief explanation]
"""
        rating, reasoning = evaluate_with_gpt(full_prompt, conversation)
        evaluation_results[f"{criteria} Rating"] = rating
        evaluation_results[f"{criteria} Reasoning"] = reasoning
    
    rating_keys = [f"{criteria} Rating" for criteria in prompts.keys()]
    ratings = [evaluation_results[key] for key in rating_keys]
    overall_score = round(sum(ratings) / len(ratings), 2) if ratings else 0
    
    evaluation_data = {
        'Evaluation Month': datetime.now().strftime("%b - %y"),
        'CSA': csa_name,
        'Ticket ID': ticket_id,
    }
    evaluation_data.update(evaluation_results)
    evaluation_data['Overall Score'] = f"{overall_score:.2f}"
    
    return evaluation_data

# Function to get already evaluated conversation IDs
def get_evaluated_conversation_ids():
   """Get list of already evaluated conversation IDs from COPSight Dump"""
   try:
       credentials = service_account.Credentials.from_service_account_file(
           SERVICE_ACCOUNT_JSON,
           scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
       )
       client = gspread.authorize(credentials)
       sheet = client.open_by_key(OUTPUT_SHEET_ID).worksheet('COPSight Dump')
       data = sheet.get_all_records()
       return set(str(row['Ticket ID']) for row in data if 'Ticket ID' in row)
   except Exception as e:
       logging.error(f"Error fetching evaluated IDs: {e}")
       return set()

# Function to process conversations in batches with real-time progress
def process_batches(conversations_df: pd.DataFrame, batch_size: int, prompts: Dict[str, str]) -> pd.DataFrame:
    try:
        loading_container = show_loading_indicator(message="Processing conversations...")
        
        evaluated_ids = get_evaluated_conversation_ids()
        conversations_df = conversations_df[~conversations_df['Ticket ID'].astype(str).isin(evaluated_ids)]

        if conversations_df.empty:
            hide_loading_indicator(loading_container)
            st.warning("No new conversations to evaluate!")
            return pd.DataFrame()

        total_conversations = len(conversations_df)
        detailed_audit_results = []
        evaluated_count = 0
        start_time = time.time()

        # Initialize session state for progress tracking
        if 'evaluated_count' not in st.session_state:
            st.session_state.evaluated_count = 0
        if 'detailed_audit_results' not in st.session_state:
            st.session_state.detailed_audit_results = []

        loading_container = st.empty()
        progress_container = st.container()

        with progress_container:
            st.markdown("### Evaluation Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()

        loading_container.image("https://i.ibb.co/zn1gJHQ/loading.gif", width=50)
        st.session_state.evaluation_running = True

        progress = 0

        for start in range(0, total_conversations, batch_size):
            if st.session_state.get("stop_requested", False):
                st.warning("Evaluation stopped by user!")
                break  # Check stop condition before starting new batch
            
            batch = conversations_df.iloc[start:start+batch_size]
            futures = []
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks for current batch
                for _, row in batch.iterrows():
                    futures.append(executor.submit(evaluate_conversation, row, prompts))
                
                # Process completed tasks
                for future in as_completed(futures):
                    if st.session_state.get("stop_requested", False):
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                    
                    try:
                        result = future.result()
                        if result:  # Only append if result is not None
                            detailed_audit_results.append(result)
                            evaluated_count += 1
                            
                            # Update session state
                            st.session_state.evaluated_count = evaluated_count
                            st.session_state.detailed_audit_results = detailed_audit_results

                            # Update progress
                            progress = evaluated_count / total_conversations
                            progress_bar.progress(progress)
                            status_text.markdown(f"**Evaluated {evaluated_count} out of {total_conversations} conversations**")

                            # Update time estimate
                            avg_time_per_conversation = (time.time() - start_time) / evaluated_count
                            remaining_conversations = total_conversations - evaluated_count
                            estimated_time_remaining = avg_time_per_conversation * remaining_conversations
                            time_text.markdown(f"**Estimated time remaining: ~{format_time(estimated_time_remaining)}**")
                    except Exception as e:
                        logging.error(f"Error processing conversation: {e}")
                        continue

    except Exception as e:
        hide_loading_indicator(loading_container)
        logging.error(f"Error in batch processing: {e}")
        st.error(f"Error processing conversations: {e}")
        return pd.DataFrame()
    finally:
        # Clean up UI and state
        progress_bar.progress(progress if st.session_state.get("stop_requested", False) else 1.0)
        status_text.markdown(f"**Evaluation {'stopped' if st.session_state.get('stop_requested', False) else 'complete'}: "
                           f"{evaluated_count} out of {total_conversations} conversations evaluated**")
        time_text.empty()
        loading_container.empty()
        
        # Reset state variables
        st.session_state.evaluation_running = False
        st.session_state.stop_requested = False

    if detailed_audit_results:
        try:
            # Write all results to Google Sheet in a single batch
            evaluation_df = pd.DataFrame(detailed_audit_results)
            write_to_google_sheet(evaluation_df, OUTPUT_SHEET_ID, 'COPSight Dump')

            sheet_url = f"https://docs.google.com/spreadsheets/d/{OUTPUT_SHEET_ID}"
            success_msg = f"""
            ✅ Evaluation {'stopped' if evaluated_count < total_conversations else 'completed'} successfully!

            - Total conversations processed: {evaluated_count}
            - Remaining conversations: {total_conversations - evaluated_count}

            View results in [COPSight Dump Sheet]({sheet_url})
            """
            st.markdown(success_msg)
            return evaluation_df
        except Exception as e:
            logging.error(f"Error writing results to Google Sheet: {e}")
            st.error(f"Error saving results to Google Sheet: {e}")
            
            # Store the evaluation results in session state for download
            st.session_state.evaluation_results = detailed_audit_results
            
            # Create a download button for the CSV
            st.warning("⚠️ Failed to save results to Google Sheet. You can download the results as CSV instead.")
            csv = pd.DataFrame(detailed_audit_results).to_csv(index=False)
            st.download_button(
                label="📥 Download Evaluation Results (CSV)",
                data=csv,
                file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the evaluation results as a CSV file"
            )
            return pd.DataFrame()

    return pd.DataFrame() 

# Function to write DataFrame to Google Sheet
def write_to_google_sheet(df: pd.DataFrame, sheet_id: str, sheet_name: str):
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_JSON,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(sheet_id).worksheet(sheet_name)
        
        # Get existing data
        existing_data = sheet.get_all_records()
        existing_df = pd.DataFrame(existing_data)
        
        # Combine existing and new data
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        
        # Convert DataFrame to list of lists
        values = [combined_df.columns.tolist()] + combined_df.values.tolist()
        
        # Clear existing content and write new data
        sheet.clear()
        sheet.update('A1', values)
        
        return True
    except Exception as e:
        logging.error(f"Error writing to Google Sheet: {e}")
        raise

# Main Streamlit UI
def main():
    st.set_page_config(
        page_title="COPSight - Conversation Evaluation",
        page_icon="👮",
        layout="wide"
    )

    st.title("👮 COPSight - Conversation Evaluation")
    st.markdown("""
    ### Welcome to COPSight!
    This tool helps evaluate customer service conversations using AI-powered analysis.
    """)

    # Initialize session state
    if 'evaluation_running' not in st.session_state:
        st.session_state.evaluation_running = False
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=50, value=10)
        max_workers = st.number_input("Max Workers", min_value=1, max_value=10, value=5)
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        if 'evaluated_count' in st.session_state:
            st.metric("Evaluated Conversations", st.session_state.evaluated_count)
        
        st.markdown("---")
        st.markdown("### 🛠️ Tools")
        if st.button("🔄 Refresh Data"):
            st.session_state.conversations_df = None
            st.session_state.conversation_count = None
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Conversation Evaluation")
        if st.button("Start Evaluation", type="primary", disabled=st.session_state.evaluation_running):
            try:
                # Fetch conversations
                conversations_df, conversation_count = fetch_conversations()
                if conversations_df is not None and not conversations_df.empty:
                    # Load prompts
                    prompts = load_prompts()
                    if prompts:
                        # Process conversations
                        process_batches(conversations_df, batch_size, prompts)
            except Exception as e:
                st.error(f"Error during evaluation: {e}")
                logging.error(f"Error during evaluation: {e}")

    with col2:
        st.header("ℹ️ Information")
        st.markdown("""
        ### How it works:
        1. Fetches conversations from Google Sheet
        2. Evaluates each conversation using AI
        3. Saves results to COPSight Dump
        4. Provides real-time progress updates
        """)

    # Stop button
    if st.session_state.evaluation_running:
        if st.button("Stop Evaluation", type="secondary"):
            st.session_state.stop_requested = True
            st.warning("Stopping evaluation... Please wait for current batch to complete.")

if __name__ == "__main__":
    main() 