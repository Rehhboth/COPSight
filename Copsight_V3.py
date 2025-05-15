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

# Add at the beginning of the file, after imports
if 'evaluation_running' not in st.session_state:
    st.session_state.evaluation_running = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Global variables
SERVICE_ACCOUNT_JSON = st.secrets["gcp_service_account"]
SHEET_ID = st.secrets["SHEET_ID"]
OUTPUT_SHEET_ID = st.secrets["OUTPUT_SHEET_ID"]
PROMPTS_SHEET_ID = st.secrets["PROMPTS_SHEET_ID"]
EXPECTED_HEADERS = ['Ticket ID', 'CSA', 'conversation_text']
API_KEY = st.secrets["OPENAI_API_KEY"]
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
SLACK_BOT_TOKEN = st.secrets["SLACK_BOT_TOKEN"]
SLACK_API_URL = "https://slack.com/api"

# Function to get credentials
def get_credentials():
    try:
        # Convert the service account JSON to a dictionary if it's a string
        if isinstance(SERVICE_ACCOUNT_JSON, str):
            service_account_info = json.loads(SERVICE_ACCOUNT_JSON)
        else:
            service_account_info = SERVICE_ACCOUNT_JSON
            
        return service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
    except Exception as e:
        logging.error(f"Error creating credentials: {e}")
        raise e

# Function to format time in a human-readable way
def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} seconds"
    minutes = math.floor(seconds / 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes} minute{'s' if minutes != 1 else ''} {remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"

# Add this function at the top of the file, after imports
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
def fetch_and_format_conversations():
    loading_container = None
    try:
        loading_container = show_loading_indicator(message="Fetching conversations from Google Sheet...")
        
        credentials = get_credentials()
        client = gspread.authorize(credentials)
        
        # Add logging for debugging
        logging.info(f"Attempting to access sheet with ID: {SHEET_ID}")
        
        try:
            spreadsheet = client.open_by_key(SHEET_ID)
            logging.info("Successfully accessed spreadsheet")
        except gspread.exceptions.SpreadsheetNotFound:
            error_msg = f"Spreadsheet not found with ID: {SHEET_ID}. Please check if the ID is correct and the service account has access."
            logging.error(error_msg)
            hide_loading_indicator(loading_container)
            st.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error accessing spreadsheet: {str(e)}"
            logging.error(error_msg)
            hide_loading_indicator(loading_container)
            st.error(error_msg)
            raise e
            
        try:
            sheet = spreadsheet.worksheet('Raw Data')
            logging.info("Successfully accessed 'Raw Data' worksheet")
        except gspread.exceptions.WorksheetNotFound:
            error_msg = "Worksheet 'Raw Data' not found in the spreadsheet. Please check if the worksheet exists."
            logging.error(error_msg)
            hide_loading_indicator(loading_container)
            st.error(error_msg)
            raise ValueError(error_msg)
            
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

        # Create a new DataFrame with required columns and proper data types
        output_df = pd.DataFrame({
            'Ticket ID': df['Ticket ID'].astype(str),
            'CSA': df['CSA'].astype(str),
            'conversation_text': df['conversation_text'].astype(str)
        })

        unique_conversation_ids = output_df['Ticket ID'].nunique()
        hide_loading_indicator(loading_container)
        return output_df, unique_conversation_ids
    except Exception as e:
        if loading_container:
            hide_loading_indicator(loading_container)
        logging.error(f"Error in fetch_and_format_conversations: {e}")
        raise e


# Function to load prompts from Google Sheet
def load_prompts():
    credentials = get_credentials()
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(PROMPTS_SHEET_ID).worksheet('Prompts')
    headers = sheet.row_values(1)  # Headers in row 1 (A1:E1)
    prompts_row = sheet.row_values(2)  # Prompts in row 2 (A2:E2)
    if len(headers) != len(prompts_row):
        raise ValueError("Mismatch between headers and prompts.")
    prompts = dict(zip(headers, prompts_row))
    return prompts


# Function to check if CSV file has expected headers
def check_csv_format(df: pd.DataFrame) -> bool:
    return all(header in df.columns for header in EXPECTED_HEADERS)


# Function to write or append evaluation results to CSV
def write_or_append_to_csv(data: Dict[str, Any], output_csv_path: str) -> None:
    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


# Function to write evaluation results to Google Sheet
def write_to_google_sheet(evaluation_data: pd.DataFrame, sheet_id: str, sheet_name: str):
    """
    Appends new evaluation data to the Google Sheet, preserving existing data.
    Drops duplicates based on 'Ticket ID', keeping the latest entry.
    """
    try:
        credentials = get_credentials()
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_key(sheet_id)
        
        # Try to get the worksheet, create it if it doesn't exist
        try:
            sheet = spreadsheet.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            sheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=26)
        
        # Read existing data
        existing_records = sheet.get_all_records()
        if existing_records:
            existing_df = pd.DataFrame(existing_records)
            # Concatenate and drop duplicates based on Ticket ID
            combined_df = pd.concat([existing_df, evaluation_data], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Ticket ID'], keep='last')
        else:
            combined_df = evaluation_data

        # Write back to sheet
        headers = combined_df.columns.tolist()
        values = [headers] + combined_df.values.tolist()
        sheet.clear()
        sheet.update('A1', values)
        logging.info(f"Successfully appended evaluation results to Google Sheet: {sheet_name}")
    except Exception as e:
        logging.error(f"Error writing to Google Sheet: {e}")
        st.error(f"Error writing to Google Sheet: {e}")


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
       credentials = get_credentials()
       client = gspread.authorize(credentials)
       sheet = client.open_by_key(OUTPUT_SHEET_ID).worksheet('COPSight Dump')
       data = sheet.get_all_records()
       return set(str(row['Ticket ID']) for row in data if 'Ticket ID' in row)
   except Exception as e:
       logging.error(f"Error fetching evaluated IDs: {e}")
       return set()


# Function to process conversations in batches with real-time progress
def process_batches(conversations_df: pd.DataFrame, batch_size: int, prompts: Dict[str, str]) -> pd.DataFrame:
    # Initialize UI elements at the start
    loading_container = None
    progress_bar = None
    status_text = None
    time_text = None
    
    try:
        loading_container = show_loading_indicator(message="Processing conversations...")
        
        evaluated_ids = get_evaluated_conversation_ids()
        conversations_df = conversations_df[~conversations_df['Ticket ID'].astype(str).isin(evaluated_ids)].copy()

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
            
            batch = conversations_df.iloc[start:start+batch_size].copy()
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
                            if progress_bar:
                                progress_bar.progress(progress)
                            if status_text:
                                status_text.markdown(f"**Evaluated {evaluated_count} out of {total_conversations} conversations**")

                            # Update time estimate
                            avg_time_per_conversation = (time.time() - start_time) / evaluated_count
                            remaining_conversations = total_conversations - evaluated_count
                            estimated_time_remaining = avg_time_per_conversation * remaining_conversations
                            if time_text:
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
        if progress_bar:
            progress_bar.progress(progress if st.session_state.get("stop_requested", False) else 1.0)
        if status_text:
            status_text.markdown(f"**Evaluation {'stopped' if st.session_state.get('stop_requested', False) else 'complete'}: "
                               f"{evaluated_count} out of {total_conversations} conversations evaluated**")
        if time_text:
            time_text.empty()
        if loading_container:
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


# Function to generate summary of reasonings using GPT
def generate_reasoning_summary(reasonings: list) -> str:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    # Limit the number of reasonings to prevent token limit issues
    MAX_REASONINGS = 50  # Adjust this number based on your needs
    if len(reasonings) > MAX_REASONINGS:
        logging.info(f"Truncating {len(reasonings)} reasonings to {MAX_REASONINGS}")
        reasonings = reasonings[:MAX_REASONINGS]
    
    # Format the reasonings into a clear prompt with truncation
    formatted_reasonings = []
    for reasoning in reasonings:
        # Truncate each reasoning to a reasonable length
        if len(reasoning) > 500:  # Adjust this number based on your needs
            reasoning = reasoning[:500] + "..."
        formatted_reasonings.append(f"- {reasoning}")
    
    formatted_reasonings_text = "\n".join(formatted_reasonings)
    
    prompt = f"""You have the following evaluation reasonings for a CSA's conversations:

{formatted_reasonings_text}

Please provide a concise summary of the evaluation, following these specific guidelines:

1. Split the feedback into two distinct sections: "Strengths" and "Areas for Improvement"
2. For each section, identify 3-5 key points
3. CRITICAL: Ensure that strengths and areas for improvement do NOT overlap or contradict each other. For example:
   - If "communication" is listed as a strength, do not list it as an area for improvement
   - If "problem-solving" is an area for improvement, do not list it as a strength
   - If "tone of voice" is mentioned positively in strengths, do not mention it in areas for improvement
4. Focus on different aspects of performance in each section
5. Keep each point specific and actionable
6. Do not mention the CSA's name or use personal pronouns
7. Format the response with clear bullet points under each section
8. Keep the total summary under 300 tokens
9. If a topic appears in both positive and negative feedback, analyze the overall trend and place it in the appropriate section
10. Ensure consistency in feedback - if a skill is generally strong, don't list minor issues with it as areas for improvement

Example format:
**Strengths:**
- Point 1
- Point 2
- Point 3

**Areas for Improvement:**
- Point 1
- Point 2
- Point 3"""

    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'system', 'content': "You are an AI assistant tasked with evaluating customer service conversations. Your role is to provide balanced, non-overlapping feedback. STRICTLY ensure that strengths and areas for improvement do not contradict each other. If a topic appears in both positive and negative feedback, analyze the overall trend and place it in the appropriate section."},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.7,
        'max_tokens': 300,
        'top_p': 1,
        'frequency_penalty': 0,
        'presence_penalty': 0
    }
    
    try:
        response = requests.post(API_URL, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                summary = result['choices'][0]['message']['content'].strip()
                
                # Verify that the summary doesn't contain contradictions
                def check_for_contradictions(summary_text):
                    strengths = summary_text.split("**Strengths:**")[1].split("**Areas for Improvement:**")[0].lower()
                    improvements = summary_text.split("**Areas for Improvement:**")[1].lower()
                    
                    # List of common aspects to check for contradictions
                    aspects = ['empathy', 'personalization', 'greeting', 'communication', 'tone', 'clarity', 'response']
                    
                    for aspect in aspects:
                        if aspect in strengths and aspect in improvements:
                            return True
                    return False
                
                # If contradictions are found, regenerate with stricter prompt
                if check_for_contradictions(summary):
                    data['messages'][0]['content'] = "You are an AI assistant tasked with evaluating customer service conversations. Your role is to provide balanced, non-overlapping feedback. STRICTLY ensure that strengths and areas for improvement do not contradict each other. If a topic appears in both positive and negative feedback, analyze the overall trend and place it in the appropriate section. DO NOT list the same aspect in both strengths and areas for improvement."
                    response = requests.post(API_URL, json=data, headers=headers, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        summary = result['choices'][0]['message']['content'].strip()
                
                return summary
            else:
                logging.error("Unexpected response format from OpenAI API")
                return "Error: Unexpected response format from OpenAI API"
        else:
            logging.error(f"OpenAI API returned status code {response.status_code}")
            return f"Error: API returned status code {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Error generating reasoning summary: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                logging.error(f"API Error Details: {error_details}")
            except:
                logging.error("Could not parse error response")
        return "Error generating summary. Please try again later."
    except Exception as e:
        logging.error(f"Unexpected error in generate_reasoning_summary: {str(e)}")
        return "An unexpected error occurred. Please try again later."


# Function to display consolidated report
def display_consolidated_report():
    try:
        loading_container = show_loading_indicator(message="Generating consolidated report...")
        
        # Authenticate and fetch data
        credentials = get_credentials()
        client = gspread.authorize(credentials)

        # Load data from Google Sheet
        sheet = client.open_by_key(OUTPUT_SHEET_ID).worksheet('COPSight Dump')
        data = sheet.get_all_records()
        df = pd.DataFrame(data)

        if df.empty:
            hide_loading_indicator(loading_container)
            st.warning("The 'COPSight Dump' sheet is empty. Please run an evaluation first.")
            return

        # Clean and prepare data
        rating_columns = [col for col in df.columns if col.endswith(' Rating')]
        df[rating_columns] = df[rating_columns].apply(pd.to_numeric, errors='coerce')
        
        # Calculate statistics
        total_audits = len(df)
        unique_csas = df['CSA'].nunique()
        overall_avg = round(df[rating_columns].mean().mean(), 2)

        # Display high-level metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Audits", f"{total_audits:,}")
        with col2:
            st.metric("Unique CSAs", f"{unique_csas:,}")
        with col3:
            st.metric("Average Overall Score", f"{overall_avg:.2f}")

        st.markdown("---")

        # Generate and display reasoning summary
        st.subheader("Evaluation Summary")
        all_reasonings = []
        reasoning_columns = [col for col in df.columns if col.endswith(' Reasoning')]
        for col in reasoning_columns:
            all_reasonings.extend(df[col].dropna().tolist())
        
        if all_reasonings:
            with st.spinner("Generating summary..."):
                summary = generate_reasoning_summary(all_reasonings)
                st.markdown(summary)
        else:
            st.info("No reasoning data available for summary generation.")

        st.markdown("---")

        # Calculate and display average scores by parameter
        st.subheader("Parameter Performance Overview")
        parameter_scores = pd.DataFrame({
            'Parameter': [col.replace(' Rating', '') for col in rating_columns],
            'Average Score': [round(df[col].mean(), 2) for col in rating_columns],
            'Fails': [(df[col] == 0).sum() for col in rating_columns]
        })
        parameter_scores = parameter_scores.sort_values('Average Score', ascending=False)
        
        st.dataframe(parameter_scores)

        # Individual CSA Analysis
        st.subheader("Individual CSA Analysis")
        selected_csa = st.selectbox("Select CSA", sorted(df['CSA'].unique()))
        
        if selected_csa:
            csa_data = df[df['CSA'] == selected_csa]
            csa_reasonings = []
            for col in reasoning_columns:
                csa_reasonings.extend(csa_data[col].dropna().tolist())
            
            if csa_reasonings:
                with st.spinner("Generating CSA summary..."):
                    csa_summary = generate_reasoning_summary(csa_reasonings)
                    st.markdown(csa_summary)
            else:
                st.info("No reasoning data available for this CSA.")

            # Show parameter-wise average scores for the CSA
            st.subheader(f"Parameter Performance for {selected_csa}")
            csa_parameter_scores = pd.DataFrame({
                'Parameter': [col.replace(' Rating', '') for col in rating_columns],
                'Average Score': [round(csa_data[col].mean(), 2) for col in rating_columns],
                'Fails': [(csa_data[col] == 0).sum() for col in rating_columns]
            })
            csa_parameter_scores = csa_parameter_scores.sort_values('Average Score', ascending=False)
            
            st.dataframe(csa_parameter_scores)

            # Show detailed ratings distribution
            st.subheader(f"Rating Distribution for {selected_csa}")
            for col in rating_columns:
                param_name = col.replace(' Rating', '')
                ratings_dist = csa_data[col].value_counts().sort_index()
                st.write(f"**{param_name}**")
                st.bar_chart(ratings_dist)

            # Export option
            if st.button("Export CSA Data"):
                csv = csa_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{selected_csa}_audit_details.csv",
                    mime="text/csv"
                )

        hide_loading_indicator(loading_container)
    except Exception as e:
        hide_loading_indicator(loading_container)
        logging.error(f"Error in consolidated report: {e}")
        st.error(f"Error generating report: {str(e)}")
        raise e


def get_associate_drive_id(associate_name):
    """Fetch Drive ID for associate from 'Associate List' in Prompts sheet (column A: name, column C: drive id)."""
    credentials = get_credentials()
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(PROMPTS_SHEET_ID).worksheet('Associate List')
    rows = sheet.get_all_values()
    # Assume first row is header, so start from second row
    for row in rows[1:]:
        # Defensive: check if row has at least 3 columns
        if len(row) >= 3:
            name = row[0].strip().lower()
            drive_id = row[2].strip()
            if name == associate_name.strip().lower():
                return drive_id
    return None



def create_scorecard_pdf(csa_data, month, csa_name):
    """Generate a PDF scorecard for the CSA matching the provided template."""
    pdf = FPDF()
    pdf.add_page()
    
    # Add logo (assuming logo.png is in the same directory)
    pdf.image("image.png", x=10, y=10, w=40)  # Adjust size and position as needed
    
    # Add "COPSight Summary" as the main header (centered)
    pdf.set_xy(0, 15)  # Start from left edge for full width centering
    pdf.set_font("Arial", "B", 20)
    pdf.cell(210, 10, "COPSight Summary", ln=True, align="C")  # 210 is full page width
    
    # Add associate information in a 2x2 grid without borders
    pdf.set_xy(10, 35)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(95, 10, f"Number of evaluations: {len(csa_data)}", ln=False, align="C")
    pdf.cell(95, 10, f"Evaluation Month: {month}", ln=True, align="C")
    
    # Second row
    pdf.set_xy(10, 45)
    pdf.cell(95, 10, f"Associate: {csa_name}", ln=False, align="C")
    # Calculate overall average score
    rating_columns = [col for col in csa_data.columns if col.endswith(' Rating')]
    overall_avg = round(csa_data[rating_columns].apply(pd.to_numeric, errors='coerce').mean().mean(), 2)
    pdf.cell(95, 10, f"Overall Score: {overall_avg:.2f}", ln=True, align="C")
    
    pdf.ln(10)  # Add some space
    
    # Table for parameter-wise scores
    pdf.set_font("Arial", "B", 12)
    pdf.cell(140, 8, "PARAMETER", 1, 0, "C")
    pdf.cell(50, 8, "AVERAGE", 1, 1, "C")
    
    pdf.set_font("Arial", "", 12)
    parameters = [
        "Greeting",
        "Issue Comprehension",
        "Tone of voice",
        "Grammar & Punctuation",
        "Closing"
    ]
    
    for i, param in enumerate(parameters):
        # Handle different column name formats
        if param == "Grammar & Punctuation":
            # Try different possible column names
            possible_columns = [
                "Grammar & Punctuation Rating",
                "Grammar and Punctuation Rating",
                "Grammar Rating",
                "Grammar & Punctuation"
            ]
            col = None
            for col_name in possible_columns:
                if col_name in csa_data.columns:
                    col = col_name
                    break
            
            if col is None:
                logging.warning(f"Could not find Grammar & Punctuation column in: {csa_data.columns.tolist()}")
                avg = 0
            else:
                try:
                    # Convert to numeric, handling any non-numeric values
                    avg = pd.to_numeric(csa_data[col].astype(str).str.strip().str.rstrip('%'), errors='coerce').mean()
                    if pd.isna(avg):
                        avg = 0
                    avg = round(avg, 2)
                except Exception as e:
                    logging.error(f"Error calculating Grammar & Punctuation average: {e}")
                    avg = 0
        else:
            col = f"{param} Rating"
            try:
                avg = pd.to_numeric(csa_data[col].astype(str).str.rstrip('%'), errors='coerce').mean() if not csa_data.empty else 0
                avg = round(avg, 2)
            except Exception as e:
                logging.error(f"Error calculating average for {param}: {e}")
                avg = 0
        
        pdf.cell(140, 8, param, 1, 0, "C")
        pdf.cell(50, 8, f"{avg:.2f}", 1, 1, "C")
    
    pdf.ln(10)  # Add some space
    
    # Overall Feedback Section with bordered tables
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "OVERALL FEEDBACK", ln=True, align="C")
    
    # Get all reasoning columns
    reasoning_columns = [col for col in csa_data.columns if col.endswith(' Reasoning')]
    all_reasonings = []
    for col in reasoning_columns:
        all_reasonings.extend(csa_data[col].dropna().tolist())
    
    if all_reasonings:
        # Generate summary using GPT
        summary = generate_reasoning_summary(all_reasonings)
        
        # Split summary into strengths and areas for improvement
        sections = summary.split('**')
        strengths = []
        improvements = []
        current_section = None
        
        for section in sections:
            if section.startswith('Strengths:'):
                current_section = strengths
            elif section.startswith('Areas for Improvement:'):
                current_section = improvements
            elif current_section is not None and section.strip():
                current_section.append(section.strip())
        
        # Helper function to add wrapped text with proper borders
        def add_wrapped_text(text, width=190):
            # Split text into sentences first
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            lines = []
            
            for sentence in sentences:
                words = sentence.split()
                current_line = []
                current_width = 0
                
                for word in words:
                    word_width = pdf.get_string_width(word + ' ')
                    if current_width + word_width <= width:
                        current_line.append(word)
                        current_width += word_width
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_width = word_width
                
                if current_line:
                    lines.append(' '.join(current_line))
            
            return lines

        # Strengths table with consistent formatting
        pdf.set_font("Arial", "B", 12)
        pdf.cell(190, 8, "STRENGTHS", 1, 1, "C")
        pdf.set_font("Arial", "", 10)  # Smaller font for content
        for strength in strengths:
            for line in strength.split('\n'):
                if line.strip().startswith('- '):
                    text = line.strip('- ').strip()
                    wrapped_lines = add_wrapped_text(text)
                    # Calculate total height needed for this point
                    total_height = len(wrapped_lines) * 8
                    # Draw border for the entire point
                    pdf.set_draw_color(0, 0, 0)  # Black border
                    pdf.rect(pdf.get_x(), pdf.get_y(), 190, total_height)
                    # Add each line of text
                    for wrapped_line in wrapped_lines:
                        pdf.cell(190, 8, wrapped_line, 0, 1, "L")  # No border for individual lines
                    pdf.ln(2)  # Add small space between points
        
        pdf.ln(5)  # Add space between tables
        
        # Areas for Improvement table with consistent formatting
        pdf.set_font("Arial", "B", 12)
        pdf.cell(190, 8, "AREAS FOR IMPROVEMENT", 1, 1, "C")
        pdf.set_font("Arial", "", 10)  # Smaller font for content
        for improvement in improvements:
            for line in improvement.split('\n'):
                if line.strip().startswith('- '):
                    text = line.strip('- ').strip()
                    wrapped_lines = add_wrapped_text(text)
                    # Calculate total height needed for this point
                    total_height = len(wrapped_lines) * 8
                    # Draw border for the entire point
                    pdf.set_draw_color(0, 0, 0)  # Black border
                    pdf.rect(pdf.get_x(), pdf.get_y(), 190, total_height)
                    # Add each line of text
                    for wrapped_line in wrapped_lines:
                        pdf.cell(190, 8, wrapped_line, 0, 1, "L")  # No border for individual lines
                    pdf.ln(2)  # Add small space between points
    else:
        pdf.set_font("Arial", "", 10)
        pdf.cell(190, 8, "No reasoning data available for summary generation.", 1, 1, "C")
    
    # Output as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return io.BytesIO(pdf_bytes)

def upload_pdf_to_drive(pdf_bytes, drive_id, filename):
    """Upload PDF to Google Drive folder using Drive ID."""
    credentials = get_credentials()
    service = build('drive', 'v3', credentials=credentials)
    file_metadata = {
        'name': filename,
        'parents': [drive_id],
        'mimeType': 'application/pdf'
    }
    media = MediaIoBaseUpload(pdf_bytes, mimetype='application/pdf')
    file = service.files().create(body=file_metadata, media_body=media, fields='id,webViewLink').execute()
    return file.get('webViewLink')

def generate_csa_scorecard():
    try:
        loading_container = show_loading_indicator(message="Generating CSA scorecard...")
        
        # Fetch COPSight Dump data
        credentials = get_credentials()
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(OUTPUT_SHEET_ID).worksheet('COPSight Dump')
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        if df.empty:
            hide_loading_indicator(loading_container)
            st.warning("No data in COPSight Dump.")
            return

        # 2. Fetch Associate List (for Drive IDs and emails)
        assoc_sheet = client.open_by_key(PROMPTS_SHEET_ID).worksheet('Associate List')
        # Get all values from the sheet
        assoc_values = assoc_sheet.get_all_values()
        # Skip header row and get data from columns A, B, and C
        assoc_data = []
        for row in assoc_values[1:]:  # Skip header row
            if len(row) >= 3:  # Ensure row has at least 3 columns
                assoc_data.append({
                    'Name': row[0].strip(),  # Column A
                    'Email': row[1].strip(),  # Column B
                    'Drive ID': row[2].strip()  # Column C
                })
        
        # Create a mapping of CSA names to their emails and drive IDs with flexible matching
        csa_info = {}
        for row in assoc_data:
            name = row['Name']
            # Store both exact and normalized versions
            csa_info[name.lower()] = {
                'email': row['Email'],
                'drive_id': row['Drive ID'],
                'original_name': name
            }
            # Also store first name only
            first_name = name.split()[0].lower()
            if first_name != name.lower():
                csa_info[first_name] = {
                    'email': row['Email'],
                    'drive_id': row['Drive ID'],
                    'original_name': name
                }

        # 3. UI: Select month, select CSA (from COPSight Dump, column 'CSA')
        months = sorted(df['Evaluation Month'].unique())
        csa_names = sorted(df['CSA'].dropna().unique())
        csa_options = ["All"] + csa_names
        month = st.selectbox("Select Month", months)
        csa = st.selectbox("Select CSA", csa_options)

        if st.button("Generate Scorecard", use_container_width=True):
            if csa == "All":
                # For each CSA, generate and upload scorecard
                for csa_name in csa_names:
                    csa_data = df[(df['CSA'] == csa_name) & (df['Evaluation Month'] == month)]
                    if csa_data.empty:
                        st.info(f"No data for {csa_name} in {month}. Skipping.")
                        continue
                    
                    # Get CSA info with flexible matching
                    csa_lower = csa_name.strip().lower()
                    csa_info_entry = None
                    
                    # Try exact match first
                    if csa_lower in csa_info:
                        csa_info_entry = csa_info[csa_lower]
                    else:
                        # Try first name match
                        first_name = csa_lower.split()[0]
                        if first_name in csa_info:
                            csa_info_entry = csa_info[first_name]
                    
                    if not csa_info_entry:
                        st.error(f"CSA info not found for {csa_name}. Please check the Associate List sheet.")
                        continue
                    
                    drive_id = csa_info_entry['drive_id']
                    email = csa_info_entry['email']
                    original_name = csa_info_entry['original_name']
                    
                    if not drive_id:
                        st.error(f"Drive ID not found for {csa_name}. Please check the Associate List sheet.")
                        continue
                        
                    pdf_bytes = create_scorecard_pdf(csa_data, month, original_name)
                    filename = f"{month.replace(' ', '')} COPSight Scorecard.pdf"
                    
                    with st.spinner(f"Uploading scorecard for {original_name}..."):
                        link = upload_pdf_to_drive(pdf_bytes, drive_id, filename)
                        
                        # Send Slack notification
                        if email:
                            slack_user_id = get_slack_user_id(email)
                            if slack_user_id:
                                if send_slack_message(slack_user_id, link, original_name):
                                    st.success(f"Slack notification sent to {original_name}")
                                else:
                                    st.warning(f"Failed to send Slack notification to {original_name}")
                            else:
                                st.warning(f"Could not find Slack user ID for {original_name}")
                        
                        st.success(f"Scorecard for {original_name} uploaded! [View in Drive]({link})")
                        st.download_button(f"Download PDF for {original_name}", data=pdf_bytes, file_name=filename, mime="application/pdf")
            else:
                csa_data = df[(df['CSA'] == csa) & (df['Evaluation Month'] == month)]
                if csa_data.empty:
                    st.warning("No data for this CSA and month.")
                    return
                    
                # Get CSA info with flexible matching
                csa_lower = csa.strip().lower()
                csa_info_entry = None
                
                # Try exact match first
                if csa_lower in csa_info:
                    csa_info_entry = csa_info[csa_lower]
                else:
                    # Try first name match
                    first_name = csa_lower.split()[0]
                    if first_name in csa_info:
                        csa_info_entry = csa_info[first_name]
                
                if not csa_info_entry:
                    st.error(f"CSA info not found for {csa}. Please check the Associate List sheet.")
                    return
                
                drive_id = csa_info_entry['drive_id']
                email = csa_info_entry['email']
                original_name = csa_info_entry['original_name']
                
                if not drive_id:
                    st.error("Drive ID not found for this associate. Please check the Associate List sheet.")
                    return
                    
                pdf_bytes = create_scorecard_pdf(csa_data, month, original_name)
                filename = f"{month.replace(' ', '')} COPSight Scorecard.pdf"
                
                with st.spinner("Uploading scorecard to Drive..."):
                    link = upload_pdf_to_drive(pdf_bytes, drive_id, filename)
                    
                    # Send Slack notification
                    if email:
                        slack_user_id = get_slack_user_id(email)
                        if slack_user_id:
                            if send_slack_message(slack_user_id, link, original_name):
                                st.success(f"Slack notification sent to {original_name}")
                            else:
                                st.warning(f"Failed to send Slack notification to {original_name}")
                        else:
                            st.warning(f"Could not find Slack user ID for {original_name}")
                    
                    st.success(f"Scorecard uploaded! [View in Drive]({link})")
                    st.download_button("Download PDF", data=pdf_bytes, file_name=filename, mime="application/pdf")

        hide_loading_indicator(loading_container)
    except Exception as e:
        hide_loading_indicator(loading_container)
        logging.error(f"Error generating CSA scorecard: {e}")
        st.error(f"Error generating scorecard: {str(e)}")

def get_slack_user_id(email: str) -> str:
    """Get Slack user ID from email address."""
    headers = {
        'Authorization': f'Bearer {SLACK_BOT_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(
            f"{SLACK_API_URL}/users.lookupByEmail?email={email}",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get('ok'):
            return data['user']['id']
        else:
            logging.error(f"Failed to get Slack user ID: {data.get('error')}")
            return None
    except Exception as e:
        logging.error(f"Error getting Slack user ID: {e}")
        return None

def send_slack_message(user_id: str, file_url: str, csa_name: str) -> bool:
    """Send Slack message with scorecard link to the user."""
    headers = {
        'Authorization': f'Bearer {SLACK_BOT_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    message = f"Hi <@{user_id}>,\n\nYour COPSight score card for the month has been uploaded. Use the following link to access it: {file_url}"
    
    payload = {
        'channel': user_id,
        'text': message
    }
    
    try:
        response = requests.post(
            f"{SLACK_API_URL}/chat.postMessage",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get('ok'):
            logging.info(f"Successfully sent Slack message to {csa_name}")
            return True
        else:
            logging.error(f"Failed to send Slack message: {data.get('error')}")
            return False
    except Exception as e:
        logging.error(f"Error sending Slack message: {e}")
        return False

# Main function to run the Streamlit app
def main() -> None:
    try:
        start_time = datetime.now()
        st.set_page_config(layout="wide", page_title="COPSight v3", page_icon="https://i.ibb.co/PYsMy0T/footer.png")
        
        # Custom CSS for improved sidebar and overall design
        st.markdown("""
            <style>
            .progress-container {
                border: 2px solid #f0f2f6;
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
                margin-bottom: 20px;
            }
            .metric-box {
                background-color: #4A90E2;
                color: #000000;
                border: 2px solid #2F80ED;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .metric-box h3 {
                margin: 0;
                font-size: 1.5em;
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                background-color: #f8f9fa;
                padding: 20px;
                border-right: 2px solid #e9ecef;
            }

            /* Footer styling */
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #000000;
                color: #E0E0E0;
                text-align: right;
                padding: 12px 20px;
                font-size: 14px;
                box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.2);
                z-index: 1000;
                font-family: Arial, sans-serif;
            }
            .footer p {
                margin: 0;
                padding: 5px 0;
                letter-spacing: 0.5px;
                display: inline-block;
            }
            .footer a {
                color: #E0E0E0;
                text-decoration: none;
                transition: color 0.3s ease;
                margin-left: 15px;
            }
            .footer a:hover {
                color: #FFFFFF;
                text-decoration: none;
            }
            .main-content {
                margin-bottom: 50px; /* Add space for fixed footer */
            }
            </style>
        """, unsafe_allow_html=True)

        # Add main content wrapper
        st.markdown('<div class="main-content">', unsafe_allow_html=True)

        # Sidebar content with improved styling
        with st.sidebar:
            # Logo
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, use_container_width=True)
            
            st.markdown("---")  # Simple separator
            
            # Menu selection with direct styling
            st.markdown('<p style="color: #1E88E5; font-size: 20px; font-weight: bold; text-align: center;">Navigation Menu</p>', unsafe_allow_html=True)
            menu_items = ["Evaluate Conversations", "Consolidated Report", "Score Card Generator"]
            choice = st.radio(" ", menu_items)
            
            st.markdown("---")  # Simple separator
            
            # Version and build time info
            st.markdown("### System Info")
            st.markdown("**Version:** 3.0.1")
            st.markdown("**Last Updated:** May 2025")

        # --- Passcode logic ---
        if 'passcode_authenticated' not in st.session_state:
            st.session_state.passcode_authenticated = False

        if choice == "Evaluate Conversations":
            st.markdown("<h1 style='text-align: center;'>COPSight</h1>", unsafe_allow_html=True)
            st.subheader("Evaluate Conversations")
            st.markdown("---")

            if not st.session_state.passcode_authenticated:
                passcode = st.text_input("Enter passcode:", type="password")
                if passcode == "156":
                    st.session_state.passcode_authenticated = True
                    st.success("✅ Access granted!")
                    st.rerun()
                elif passcode:
                    st.warning("⚠️ Please enter the correct passcode to proceed with evaluation.")
                return

            try:
                # Always fetch fresh data from Google Sheet
                output_df, unique_conversation_count = fetch_and_format_conversations()

                # Get count of new conversations
                evaluated_ids = get_evaluated_conversation_ids()
                new_conversations = output_df[~output_df['Ticket ID'].astype(str).isin(evaluated_ids)]
                new_count = len(new_conversations)

                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<div class='metric-box'><h3>Total Conversations: {unique_conversation_count}</h3></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-box'><h3>New Conversations: {new_count}</h3></div>", unsafe_allow_html=True)

                st.markdown("")  # Add a blank line for spacing

                # --- Start/Stop Evaluation Button Logic ---
                button_col = st.columns([1, 2, 1])[1]  # Center the button
                with button_col:
                    if not st.session_state.evaluation_running:
                        if st.button("🚀 Start Evaluation", key="start_eval", help="Begin the evaluation process", use_container_width=True):
                            st.session_state.evaluation_running = True
                            st.session_state.stop_requested = False
                            st.session_state.eval_results = []
                            st.rerun()
                    else:
                        if st.button("⏹️ Stop Evaluation", key="stop_eval", help="Stop the current evaluation process", use_container_width=True):
                            st.session_state.stop_requested = True
                            st.info("⏳ Stopping evaluations...")

                # --- Evaluation Process ---
                if st.session_state.evaluation_running:
                    prompts = load_prompts()
                    process_batches(new_conversations, BATCH_SIZE, prompts)

            except Exception as e:
                st.error(f"Error fetching or processing conversations: {e}")
                logging.error(f"Error fetching or processing conversations: {e}")

        elif choice == "Consolidated Report":
            st.markdown("<h1 style='text-align: center;'>COPSight Report</h1>", unsafe_allow_html=True)
            st.subheader("Consolidated Report")
            st.markdown("---")
            display_consolidated_report()

        elif choice == "Score Card Generator":
            generate_csa_scorecard()

        # Footer with improved styling
        footer = """
        <div class="footer">
            <p>© 2025 COPSight <i>v3</i> | Powered by BI Team</p>
        </div>
        """
        st.markdown(footer, unsafe_allow_html=True)

        # Close main content wrapper
        st.markdown('</div>', unsafe_allow_html=True)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        st.sidebar.info(f"⚡ Page built in {execution_time:.2f} seconds")

    except Exception as e:
        logging.error(f"Error in main function: {e}")
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

