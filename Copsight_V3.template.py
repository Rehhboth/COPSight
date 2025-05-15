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
SERVICE_ACCOUNT_JSON = "path/to/your/service-account.json"  # Replace with your service account JSON path
SHEET_ID = "your-sheet-id"  # Replace with your Google Sheet ID
OUTPUT_SHEET_ID = "your-output-sheet-id"  # Replace with your output sheet ID
PROMPTS_SHEET_ID = "your-prompts-sheet-id"  # Replace with your prompts sheet ID
EXPECTED_HEADERS = ['Ticket ID', 'CSA', 'conversation_text']
API_KEY = "your-openai-api-key"  # Replace with your OpenAI API key
API_URL = 'https://api.openai.com/v1/chat/completions'
MAX_WORKERS = 5
RATE_LIMIT = 60
BATCH_SIZE = 10

# Add Slack configuration
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN', '')  # Set this via environment variable
SLACK_API_URL = "https://slack.com/api"

# Rest of your code here... 