import sys
import os
import io
import logging
from PIL import Image
import requests
from io import BytesIO
import logging
import hashlib
from pathlib import Path
import plotly.graph_objects as go
import re

# Set up more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# Add the parent directory (project root) to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Rest of your imports
import streamlit as st
import sqlite3
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import altair as alt
from datetime import timedelta
import logging
from functools import partial
import json
from io import BytesIO
import nltk
from streamlit_navigation_bar import st_navbar

# Import from the same directory
from sample_data import get_sample_feedback
from database import insert_feedback
from utils import setup_keyboard_shortcuts, get_staff_suggestions

# Add this at the top of app.py with other imports
import logging
logger = logging.getLogger(__name__)

def get_favicon():
    """Get favicon from local storage or download if not available."""
    # Create assets directory if it doesn't exist
    assets_dir = Path(project_root) / "assets"
    try:
        assets_dir.mkdir(exist_ok=True)
        logging.info(f"Assets directory confirmed at: {assets_dir}")
    except Exception as e:
        logging.error(f"Failed to create assets directory: {e}")
        return None
    
    favicon_path = assets_dir / "methodist_favicon.ico"
    # Updated URL to the correct Methodist favicon
    favicon_urls = [
        "https://bestcare.org/favicon.ico",
        "https://www.bestcare.org/favicon.ico",
        "https://bestcare.org/sites/default/files/favicon.ico",
        "https://bestcare.org/themes/methodist/favicon.ico"
    ]
    
    # First try to load local file if it exists
    if favicon_path.exists():
        try:
            favicon = Image.open(favicon_path)
            logging.info("Successfully loaded existing favicon")
            return favicon
        except Exception as e:
            logging.error(f"Failed to open existing favicon: {e}")
            favicon_path.unlink(missing_ok=True)
            logging.info("Deleted corrupted favicon file")
    
    # Try each URL until one works
    for url in favicon_urls:
        try:
            logging.info(f"Attempting to download favicon from {url}")
            response = requests.get(
                url,
                timeout=5,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'image/x-icon,image/*',
                },
                verify=True
            )
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type or 'icon' in content_type:
                    try:
                        # Save the favicon locally
                        with open(favicon_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Try to open the saved file
                        favicon = Image.open(favicon_path)
                        logging.info(f"Successfully downloaded and saved favicon from {url}")
                        return favicon
                    except Exception as e:
                        logging.error(f"Failed to save or process favicon from {url}: {e}")
                        continue
                else:
                    logging.warning(f"Invalid content type from {url}: {content_type}")
            else:
                logging.warning(f"Failed to download from {url}: HTTP {response.status_code}")
                if response.status_code == 404:
                    logging.error(f"404 Response text: {response.text[:200]}")
                
        except requests.RequestException as e:
            logging.warning(f"Request failed for {url}: {e}")
            continue
        except Exception as e:
            logging.warning(f"Unexpected error trying {url}: {e}")
            continue
    
    logging.error("Failed to get favicon from any URL")
    return None

# Move the page config setup to after logging is configured
try:
    favicon = get_favicon()
    if favicon:
        st.set_page_config(
            page_title="Hospital Stay Feedback",
            page_icon=favicon,
            layout="wide"
        )
        logging.info("Successfully set page config with favicon")
    else:
        st.set_page_config(
            page_title="Hospital Stay Feedback",
            layout="wide"
        )
        logging.warning("Using default page config without favicon")
except Exception as e:
    logging.error(f"Error setting page config: {e}")
    st.set_page_config(
        page_title="Hospital Stay Feedback",
        layout="wide"
    )

# Update DB_FILE path to be relative to project root
DB_FILE = os.path.join(project_root, "feedback.db")

# Setup function to ensure necessary resources are available
def setup_environment():
    # Download NLTK data
    nltk.download('vader_lexicon')

    # Download SpaCy model if not already available
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")

# Call setup function
setup_environment()

# Initialize SpaCy and NLTK Sentiment Analyzer
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Cache the database connection and queries
def get_database_connection():
    try:
        return sqlite3.connect(DB_FILE)
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        st.error("Failed to connect to the database. Please try again later.")
        return None

@st.cache_data
def get_filtered_feedback(_conn, filters):
    try:
        query = "SELECT * FROM feedback"
        df = pd.read_sql_query(query, _conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Apply filters
        for column, values in filters.items():
            if values:
                df = df[df[column].isin(values)]
        
        return df
    except Exception as e:
        logging.error(f"Error fetching feedback: {e}")
        st.error("An error occurred while retrieving the feedback data.")
        return pd.DataFrame()

def check_db_exists():
    """Check if the database exists and has data."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feedback")
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except sqlite3.Error:
        return False

def init_db(conn):
    """Initialize the database only if it doesn't exist or is empty."""
    try:
        # First check if the table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Check if the table has data
            cursor.execute("SELECT COUNT(*) FROM feedback")
            has_data = cursor.fetchone()[0] > 0
            if has_data:
                logging.info("Database already initialized with data.")
                return

        # If we get here, we need to initialize the database
        logging.info("Initializing database with sample data...")
        
        # Create a new feedback table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                room_number TEXT,
                stay_feedback TEXT,
                call_button_response TEXT,
                bathroom_help_frequency TEXT,
                nurse_explanation_clarity TEXT,
                nurse_listening TEXT,
                nurse_courtesy TEXT,
                recognition TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()

        # Insert sample data only if table is empty
        cursor.execute("SELECT COUNT(*) FROM feedback")
        if cursor.fetchone()[0] == 0:
            sample_feedback = get_sample_feedback()
            cursor.executemany("""
                INSERT INTO feedback (
                    room_number, stay_feedback, call_button_response,
                    bathroom_help_frequency, nurse_explanation_clarity,
                    nurse_listening, nurse_courtesy, recognition, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, sample_feedback)
            conn.commit()
            logging.info(f"Inserted {len(sample_feedback)} sample feedback entries.")
        
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
        raise

def insert_feedback(conn, data):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (
            room_number, stay_feedback, call_button_response,
            bathroom_help_frequency, nurse_explanation_clarity,
            nurse_listening, nurse_courtesy, recognition, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def get_all_feedback(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM feedback")
    return cursor.fetchall()

def plot_response_distribution(df, column, title):
    """Create a bar chart showing response distribution."""
    if df.empty:
        return None
    
    # Group by the specified column and count the occurrences
    response_counts = df.groupby(column).size().reset_index(name='counts')
    
    # Create an Altair chart for the response distribution
    chart = alt.Chart(response_counts).mark_bar().encode(
        x=alt.X(f'{column}:N', title=title),
        y=alt.Y('counts:Q', title='Count'),
        color=alt.Color(f'{column}:N', legend=None)
    ).properties(
        width=500,
        height=300,
        title=title
    )
    return chart

def plot_trend_analysis(df, column, window=7):
    """Create a line chart showing trends over time."""
    if df.empty:
        return None
    
    # Group by date and calculate daily counts
    daily_counts = df.groupby(df['timestamp'].dt.date)[column].value_counts().unstack(fill_value=0)
    
    # Calculate rolling average
    rolling_avg = daily_counts.rolling(window=window, min_periods=1).mean()
    
    # Create the plot
    fig = px.line(
        rolling_avg,
        title=f'{column} Trends ({window}-day rolling average)',
        labels={'value': 'Count', 'index': 'Date'}
    )
    return fig

def generate_sentiment_wordcloud(text, feedback_df, category=None, start_date=None, end_date=None):
    """
    Generate an enhanced word cloud focused on managerial insights.
    
    Args:
        text (str): Combined feedback text
        feedback_df (pd.DataFrame): DataFrame containing all feedback data
        category (str, optional): Specific feedback category to focus on
        start_date (datetime, optional): Start date for trend analysis
        end_date (datetime, optional): End date for trend analysis
    """
    doc = nlp(text)
    
    # Categorized keywords for different aspects of hospital service
    keywords = {
        'responsiveness': {
            'delay', 'slow', 'quick', 'wait', 'response', 'immediate', 'fast',
            'button', 'call', 'urgent', 'emergency', 'attention'
        },
        'comfort': {
            'noise', 'quiet', 'clean', 'dirty', 'cold', 'hot', 'comfortable',
            'uncomfortable', 'sleep', 'rest', 'bed', 'temperature', 'smell'
        },
        'care_quality': {
            'pain', 'medication', 'treatment', 'care', 'attention', 'help',
            'assist', 'support', 'explain', 'understand', 'clear', 'confused'
        },
        'staff_behavior': {
            'rude', 'kind', 'polite', 'friendly', 'helpful', 'professional',
            'attentive', 'listening', 'caring', 'respectful', 'courteous'
        }
    }
    
    # Flatten keywords for easy lookup
    all_keywords = {word for category in keywords.values() for word in category}
    
    # Extract bigrams and trigrams
    def extract_ngrams(doc, n):
        return [
            '_'.join([token.text.lower() for token in doc[i:i+n]])
            for i in range(len(doc)-n+1)
            if any(token.text.lower() in all_keywords for token in doc[i:i+n])
        ]
    
    bigrams = extract_ngrams(doc, 2)
    trigrams = extract_ngrams(doc, 3)
    
    # Process terms with context
    terms = []
    term_categories = {}
    term_sentiments = {}
    
    # Process individual words, bigrams, and trigrams
    for term in set(bigrams + trigrams + [token.text.lower() for token in doc if token.is_alpha]):
        # Skip if term doesn't contain any keywords
        if not any(keyword in term for keyword in all_keywords):
            continue
        
        # Find category for term
        for cat, cat_keywords in keywords.items():
            if any(keyword in term for keyword in cat_keywords):
                term_categories[term] = cat
                break
        
        # Get context and sentiment
        term_instances = [i for i, t in enumerate(doc) if term in t.text.lower()]
        term_contexts = []
        
        for idx in term_instances:
            start = max(0, idx - 5)
            end = min(len(doc), idx + 6)
            context = doc[start:end].text
            term_contexts.append(context)
        
        if term_contexts:
            # Calculate average sentiment with emphasis on negative feedback
            sentiments = [sentiment_analyzer.polarity_scores(ctx)['compound'] for ctx in term_contexts]
            avg_sentiment = sum(sentiments) / len(sentiments)
            # Amplify negative sentiments
            if avg_sentiment < 0:
                avg_sentiment *= 1.5
            term_sentiments[term] = avg_sentiment
            terms.append(term)
    
    # Calculate term frequencies with trend analysis if dates provided
    term_frequencies = {}
    if start_date and end_date:
        # Convert date to datetime for comparison
        start_datetime = pd.Timestamp(start_date)
        end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        
        date_mask = (feedback_df['timestamp'] >= start_datetime) & (feedback_df['timestamp'] <= end_datetime)
        recent_feedback = feedback_df[date_mask]
        
        for term in terms:
            # Count occurrences in recent feedback
            freq = sum(recent_feedback['stay_feedback'].str.contains(term, case=False, na=False))
            # Weight recent issues more heavily
            term_frequencies[term] = freq * 1.5
    else:
        term_frequencies = {term: text.lower().count(term) for term in terms}
    
    # Adjust frequencies based on sentiment and category
    for term in terms:
        base_freq = term_frequencies.get(term, 1)
        sentiment = term_sentiments.get(term, 0)
        category = term_categories.get(term, 'other')
        
        # Increase weight for negative sentiments and certain categories
        category_weights = {
            'responsiveness': 1.3,
            'care_quality': 1.2,
            'comfort': 1.1,
            'staff_behavior': 1.0
        }
        
        weight = category_weights.get(category, 1.0)
        if sentiment < 0:
            weight *= 1.5
        
        term_frequencies[term] = base_freq * weight
    
    if not term_frequencies:
        # If no terms were found, create a simple wordcloud with the original text
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            max_words=50,
            min_font_size=10,
            max_font_size=80
        ).generate(text)
        return wordcloud, {}, {}
    
    # Color function based on sentiment and category
    def color_func(word, **kwargs):
        sentiment = term_sentiments.get(word.lower(), 0)
        category = term_categories.get(word.lower(), 'other')
        
        # Category-based colors with sentiment variation
        category_colors = {
            'responsiveness': ('#FF0000', '#FFB6C1'),  # Red spectrum
            'care_quality': ('#0000FF', '#ADD8E6'),    # Blue spectrum
            'comfort': ('#008000', '#90EE90'),         # Green spectrum
            'staff_behavior': ('#800080', '#DDA0DD'),  # Purple spectrum
        }
        
        if category in category_colors:
            negative_color, positive_color = category_colors[category]
            return negative_color if sentiment < 0 else positive_color
        return '#808080'  # Gray for uncategorized terms
    
    # Generate the word cloud
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        color_func=color_func,
        max_words=50,
        min_font_size=10,
        max_font_size=80,
        prefer_horizontal=0.7,
        collocations=False
    ).generate_from_frequencies(term_frequencies)
    
    return wordcloud, term_categories, term_sentiments

def web_form_page(conn):
    st.subheader("Patient Feedback Form")
    
    with st.form(key='feedback_form', clear_on_submit=True):
        # Room number with validation
        st.markdown("### Room Number")
        room_number = st.text_input(
            label="Enter 3-digit room number (Tab ↹)",
            placeholder="123",
            max_chars=3,
            help="Press Tab to move to next field",
            key="room_number_input"
        )

        # Quick-select options
        st.markdown("### Quick Response Selection")
        quick_select = st.radio(
            "Set all responses to:",
            options=["Custom", "All Always", "All Usually", "All Sometimes"],
            horizontal=True,
            key="quick_select"
        )

        if quick_select != "Custom":
            selected_value = quick_select.split(" ")[1]
            st.session_state.update({
                'call_button_response': selected_value,
                'bathroom_help_frequency': selected_value,
                'nurse_explanation_clarity': selected_value,
                'nurse_listening': selected_value,
                'nurse_courtesy': selected_value
            })

        # Sequential questions with radio + text input
        st.markdown("### Patient Feedback")
        
        # Question 1
        st.markdown("#### 1. How has your stay been so far?")
        stay_feedback = st.text_area(
            label="Stay feedback",  # Add proper label
            label_visibility="collapsed",  # Hide the label but maintain accessibility
            placeholder="Enter patient feedback...",
            key="stay_feedback",
            help="Press Alt+S to focus",
            max_chars=500,
            height=100
        )

        # Question 2
        st.markdown("#### 2. Call Button Response Time")
        call_button_response = st.radio(
            "After pressing call button, how often was help received promptly?",
            options=["Always", "Usually", "Sometimes", "Never"],
            horizontal=True,
            key="call_button_response"
        )
        call_button_custom = st.text_input(
            "Additional comments about call button response (optional):",
            key="call_button_custom"
        )

        # Question 3
        st.markdown("#### 3. Bathroom Assistance")
        bathroom_help_frequency = st.radio(
            "How often did you get help with bathroom needs promptly?",
            options=["Always", "Usually", "Sometimes", "Never"],
            horizontal=True,
            key="bathroom_help"
        )
        bathroom_help_custom = st.text_input(
            "Additional comments about bathroom assistance (optional):",
            key="bathroom_help_custom"
        )

        # Question 4
        st.markdown("#### 4. Nurse Communication")
        nurse_explanation_clarity = st.radio(
            "How often did nurses explain things clearly?",
            options=["Always", "Usually", "Sometimes", "Never"],
            horizontal=True,
            key="clarity"
        )
        nurse_explanation_custom = st.text_input(
            "Additional comments about nurse explanations (optional):",
            key="clarity_custom"
        )

        # Question 5
        st.markdown("#### 5. Nurse Listening")
        nurse_listening = st.radio(
            "How often did nurses listen carefully to you?",
            options=["Always", "Usually", "Sometimes", "Never"],
            horizontal=True,
            key="listening"
        )
        nurse_listening_custom = st.text_input(
            "Additional comments about nurse listening (optional):",
            key="listening_custom"
        )

        # Question 6
        st.markdown("#### 6. Nurse Courtesy")
        nurse_courtesy = st.radio(
            "How often were nurses courteous and respectful?",
            options=["Always", "Usually", "Sometimes", "Never"],
            horizontal=True,
            key="courtesy"
        )
        nurse_courtesy_custom = st.text_input(
            "Additional comments about nurse courtesy (optional):",
            key="courtesy_custom"
        )

        # Question 7
        st.markdown("#### 7. Staff Recognition")
        recognition = st.text_area(
            label="Staff recognition",  # Add proper label
            label_visibility="collapsed",  # Hide the label but maintain accessibility
            placeholder="Enter staff names... (Type @ for suggestions)",
            key="recognition",
            help="Type @ to see staff name suggestions"
        )

        # Staff name suggestions
        if recognition and '@' in recognition:
            suggestions = get_staff_suggestions(recognition)
            if suggestions:
                selected_staff = st.selectbox(
                    "Select staff member to add:",
                    suggestions,
                    key="staff_select"
                )
                if selected_staff:
                    recognition = recognition.rsplit('@', 1)[0] + selected_staff + " "

        # Submit button
        submit_button = st.form_submit_button(
            label='Submit Feedback (Ctrl+Enter)',
            use_container_width=True
        )

        if submit_button:
            if room_number.isdigit() and len(room_number) == 3:
                try:
                    # Handle empty custom inputs
                    call_button_custom = call_button_custom.strip() if call_button_custom else ''
                    bathroom_help_custom = bathroom_help_custom.strip() if bathroom_help_custom else ''
                    nurse_explanation_custom = nurse_explanation_custom.strip() if nurse_explanation_custom else ''
                    nurse_listening_custom = nurse_listening_custom.strip() if nurse_listening_custom else ''
                    nurse_courtesy_custom = nurse_courtesy_custom.strip() if nurse_courtesy_custom else ''
                    
                    # Combine radio selections with custom input
                    feedback_data = (
                        room_number,
                        stay_feedback if stay_feedback else '',
                        f"{call_button_response}{' ' + call_button_custom if call_button_custom else ''}",
                        f"{bathroom_help_frequency}{' ' + bathroom_help_custom if bathroom_help_custom else ''}",
                        f"{nurse_explanation_clarity}{' ' + nurse_explanation_custom if nurse_explanation_custom else ''}",
                        f"{nurse_listening}{' ' + nurse_listening_custom if nurse_listening_custom else ''}",
                        f"{nurse_courtesy}{' ' + nurse_courtesy_custom if nurse_courtesy_custom else ''}",
                        recognition if recognition else '',
                        datetime.now().isoformat()
                    )
                    
                    # Insert feedback with improved error handling
                    success, message = insert_feedback(conn, feedback_data)
                    if success:
                        st.success(f"✅ {message} (Press Alt+N for new form)")
                        # Clear the form
                        st.cache_data.clear()
                        st.session_state.clear()
                        st.session_state.refresh_data = True
                    else:
                        st.error(message)
                        # Show specific guidance based on the error
                        if "Required fields missing" in message:
                            st.warning("Please ensure all radio button selections are made.")
                        elif "Database integrity error" in message:
                            st.warning("This may be a duplicate entry. Please check the room number and timestamp.")
                        
                except Exception as e:
                    error_msg = (
                        "An error occurred while processing your feedback. "
                        f"Error details: {str(e)}\n"
                        "Please try again or contact support if the problem persists."
                    )
                    logger.error(error_msg, exc_info=True)
                    st.error(error_msg)
                    
                    # Show technical details in an expander for debugging
                    with st.expander("Technical Details"):
                        st.code(f"Error Type: {type(e).__name__}\n{str(e)}")
            else:
                st.error("Room number must be a 3-digit number (e.g., 123)")
                st.info("Please enter exactly three digits for the room number.")

# Add this CSS to enable word wrap in the dataframe
def add_table_styles():
    return """
        <style>
            .dataframe td {
                white-space: normal !important;
                max-width: 200px;
                word-wrap: break-word;
            }
            .dataframe th {
                white-space: normal !important;
                max-width: 200px;
                word-wrap: break-word;
            }
            .stDataFrame {
                width: 100%;
            }
        </style>
    """

# Update the reports page function
def reports_page(conn):
    st.subheader("Reports")
    
    try:
        @st.cache_data(ttl=300)
        def load_feedback_data(_conn):
            query = "SELECT * FROM feedback ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, _conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        # Clear cache if needed
        if st.session_state.get('refresh_data'):
            st.cache_data.clear()
            st.session_state.refresh_data = False
        
        try:
            df = load_feedback_data(conn)
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}", exc_info=True)
            st.error("Failed to load feedback data. Please try again.")
            return
        
        if df.empty:
            st.warning("No feedback data available.")
            return

        # Date Range Filter with Presets - Default to Last 7 Days
        st.subheader("Date Range Filter")
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
        
        with filter_col1:
            preset = st.selectbox(
                "Quick Select",
                ["Last 7 Days", "Last 30 Days", "Last Quarter", "Custom"],
                index=0  # Set default to "Last 7 Days"
            )
        
        with filter_col2:
            if preset == "Custom":
                start_date = st.date_input(
                    "Start Date",
                    value=df['timestamp'].min().date(),
                    min_value=df['timestamp'].min().date(),
                    max_value=df['timestamp'].max().date()
                )
            else:
                days_lookup = {"Last 7 Days": 7, "Last 30 Days": 30, "Last Quarter": 90}
                start_date = datetime.now().date() - timedelta(days=days_lookup.get(preset, 7))
        
        with filter_col3:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                min_value=df['timestamp'].min().date(),
                max_value=df['timestamp'].max().date()
            )

        # Apply date filter
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        filtered_df = df[mask].copy()

        # Enhanced Trend Analysis
        st.subheader("Performance Trends")
        st.markdown("""
        ### Understanding the Trends
        - Each line represents a different aspect of care
        - Higher scores (closer to "Always") indicate better performance
        - The rolling average smooths daily variations
        - Annotations highlight significant changes
        """)

        # Calculate date range for proper scaling
        date_range = filtered_df['timestamp'].dt.date.unique()
        if len(date_range) > 1:
            min_date = min(date_range)
            max_date = max(date_range)
            date_diff = (max_date - min_date).days
            if date_diff < 7:
                min_date = max_date - timedelta(days=7)
        else:
            min_date = date_range[0] - timedelta(days=3)
            max_date = date_range[0] + timedelta(days=3)

        try:
            # Create trends visualization
            metrics = {
                'Call Button Response': 'call_button_response',
                'Bathroom Assistance': 'bathroom_help_frequency',
                'Clear Explanations': 'nurse_explanation_clarity',
                'Active Listening': 'nurse_listening',
                'Staff Courtesy': 'nurse_courtesy'
            }
            
            fig = go.Figure()
            
            for metric_name, column in metrics.items():
                daily_scores = filtered_df.groupby(filtered_df['timestamp'].dt.date)[column].agg(
                    lambda x: x.map({'Always': 4, 'Usually': 3, 'Sometimes': 2, 'Never': 1}).mean()
                ).rolling(7, min_periods=1).mean()
                
                fig.add_trace(go.Scatter(
                    x=daily_scores.index,
                    y=daily_scores.values,
                    name=metric_name,
                    mode='lines+markers',
                    hovertemplate='%{y:.2f}<extra>' + metric_name + '</extra>'
                ))

            fig.update_layout(
                title='Care Quality Metrics Over Time (7-Day Rolling Average)',
                xaxis_title='Date',
                yaxis_title='Response Score',
                yaxis=dict(
                    ticktext=['Never', 'Sometimes', 'Usually', 'Always'],
                    tickvals=[1, 2, 3, 4],
                    range=[0.5, 4.5]
                ),
                hovermode='x unified',
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button for trends chart
            st.download_button(
                label="📥 Download Trends Chart",
                data=fig.to_image(format="png"),
                file_name=f"trends_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png",
                mime="image/png"
            )

        except Exception as e:
            logger.error(f"Error creating trends visualization: {e}", exc_info=True)
            st.error("Failed to create trends visualization. Please try again.")

        # Add Insights Section
        st.subheader("Performance Insights")
        
        try:
            # Calculate insights
            total_responses = len(filtered_df)
            metrics_analysis = {}
            
            for metric_name, column in metrics.items():
                always_count = len(filtered_df[filtered_df[column] == 'Always'])
                usually_count = len(filtered_df[filtered_df[column] == 'Usually'])
                sometimes_count = len(filtered_df[filtered_df[column] == 'Sometimes'])
                never_count = len(filtered_df[filtered_df[column] == 'Never'])
                
                metrics_analysis[metric_name] = {
                    'always_pct': (always_count / total_responses * 100),
                    'usually_pct': (usually_count / total_responses * 100),
                    'sometimes_pct': (sometimes_count / total_responses * 100),
                    'never_pct': (never_count / total_responses * 100)
                }
            
            # Display insights
            st.markdown("### Key Findings")
            
            # Call Button Response
            st.markdown("#### 🔔 Response Time Analysis")
            call_stats = metrics_analysis['Call Button Response']
            st.markdown(f"""
            - {call_stats['always_pct']:.1f}% report consistently prompt responses
            - {call_stats['sometimes_pct'] + call_stats['never_pct']:.1f}% experience delays
            - Most common response pattern: {filtered_df['call_button_response'].mode().iloc[0]}
            """)

            # Bathroom Assistance
            st.markdown("#### 🚽 Bathroom Assistance Analysis")
            bath_stats = metrics_analysis['Bathroom Assistance']
            st.markdown(f"""
            - {bath_stats['always_pct']:.1f}% receive immediate assistance
            - {bath_stats['usually_pct']:.1f}% report generally timely help
            - {bath_stats['sometimes_pct'] + bath_stats['never_pct']:.1f}% experience inconsistent support
            """)

            # Communication Quality
            st.markdown("#### 💬 Communication Effectiveness")
            explain_stats = metrics_analysis['Clear Explanations']
            listen_stats = metrics_analysis['Active Listening']
            st.markdown(f"""
            - Clear Explanations: {explain_stats['always_pct']:.1f}% consistently clear
            - Active Listening: {listen_stats['always_pct']:.1f}% feel fully heard
            - Combined Excellence: {((filtered_df['nurse_explanation_clarity'] == 'Always') & (filtered_df['nurse_listening'] == 'Always')).mean() * 100:.1f}% report optimal communication
            """)

            # Staff Courtesy
            st.markdown("#### 👥 Staff Interaction Quality")
            courtesy_stats = metrics_analysis['Staff Courtesy']
            recognition_rate = len(filtered_df[filtered_df['recognition'].str.len() > 0]) / len(filtered_df) * 100
            st.markdown(f"""
            - {courtesy_stats['always_pct']:.1f}% report consistently courteous interactions
            - {courtesy_stats['usually_pct'] + courtesy_stats['always_pct']:.1f}% positive interaction rate
            - {recognition_rate:.1f}% of feedback includes staff recognition
            """)

            # Overall Service Quality
            st.markdown("#### 📊 Overall Service Assessment")
            overall_always = sum(m['always_pct'] for m in metrics_analysis.values()) / len(metrics_analysis)
            st.markdown(f"""
            - Overall Excellence Rate: {overall_always:.1f}% across all metrics
            - Most Consistent Area: {max(metrics_analysis.items(), key=lambda x: x[1]['always_pct'])[0]}
            - Primary Focus Area: {max(metrics_analysis.items(), key=lambda x: x[1]['sometimes_pct'] + x[1]['never_pct'])[0]}
            """)

        except Exception as e:
            logger.error(f"Error generating insights: {e}", exc_info=True)
            st.error("Failed to generate insights. Please try again.")

        # Word Cloud Section
        st.subheader("Feedback Analysis")
        
        # Generate word cloud data
        if not filtered_df.empty:
            # Process text to extract meaningful phrases
            all_text = ' '.join([
                str(text) for text in filtered_df['stay_feedback'] 
                if isinstance(text, str)
            ])
            
            # Define meaningful phrase patterns with sentiment handling
            PHRASE_PATTERNS = {
                'cleanliness': [
                    (r'(?:was|is|were|are)\s+(?:very\s+)?clean', 'positive'),
                    (r'(?:was|is|were|are)\s+not\s+(?:very\s+)?clean', 'negative'),
                    (r'(?:was|is|were|are)\s+(?:very\s+)?dirty', 'negative'),
                ],
                'food': [
                    (r'(?:food|meal|meals)\s+(?:was|were|is|are)\s+(?:very\s+)?(?:good|great|excellent|delicious)', 'positive'),
                    (r'(?:food|meal|meals)\s+(?:was|were|is|are)\s+(?:very\s+)?(?:bad|poor|cold|terrible)', 'negative'),
                ],
                'staff': [
                    (r'(?:staff|nurses|nurse|doctors|doctor)\s+(?:was|were|are|is)\s+(?:very\s+)?(?:helpful|friendly|professional|caring|attentive|responsive)', 'positive'),
                    (r'(?:staff|nurses|nurse|doctors|doctor)\s+(?:was|were|are|is)\s+not\s+(?:very\s+)?(?:helpful|friendly|professional|caring|attentive|responsive)', 'negative'),
                    (r'(?:staff|nurses|nurse|doctors|doctor)\s+(?:was|were|are|is)\s+(?:very\s+)?(?:rude|unhelpful|unprofessional)', 'negative'),
                ]
            }
            
            # Extract complete phrases with sentiment
            phrase_freq = {}
            phrase_sentiments = {}
            
            for category, patterns in PHRASE_PATTERNS.items():
                for pattern, sentiment_type in patterns:
                    matches = re.finditer(pattern, all_text.lower())
                    for match in matches:
                        phrase = match.group(0)
                        if len(phrase.split()) >= 2:
                            count = all_text.lower().count(phrase)
                            if sentiment_type == 'negative' and 'not' not in phrase:
                                phrase = 'not ' + phrase
                            phrase_freq[phrase] = count
                            phrase_sentiments[phrase] = 1 if sentiment_type == 'positive' else -1
            
            # Create word cloud with sentiment-aware coloring
            if phrase_freq:
                def color_func(word, **kwargs):
                    sentiment = phrase_sentiments.get(word, 0)
                    return '#2ecc71' if sentiment > 0 else '#e74c3c' if sentiment < 0 else '#f1c40f'
                
                wordcloud = WordCloud(
                    width=1200,
                    height=600,
                    background_color='white',
                    color_func=color_func,
                    max_words=100,
                    collocations=True,
                    prefer_horizontal=0.7
                ).generate_from_frequencies(phrase_freq)
                
                # Display word cloud with download button
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    fig_wordcloud = plt.figure(figsize=(20, 10))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(fig_wordcloud)
                    plt.close(fig_wordcloud)
                    
                    # Word cloud download button
                    img_buf = BytesIO()
                    wordcloud.to_image().save(img_buf, format='PNG')
                    st.download_button(
                        label="📥 Download Word Cloud",
                        data=img_buf.getvalue(),
                        file_name=f"wordcloud_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png",
                        mime="image/png"
                    )
                
                with col2:
                    # Top phrases with sentiment analysis
                    st.markdown("### Top Phrases")
                    top_phrases = dict(sorted(phrase_freq.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:10])
                    for phrase, count in top_phrases.items():
                        sentiment = phrase_sentiments.get(phrase, 0)
                        emoji = "🟢" if sentiment > 0 else "🔴" if sentiment < 0 else "🟡"
                        st.markdown(f"{emoji} {phrase} ({count})")

        # Detailed Feedback View
        st.subheader("Detailed Feedback")
        
        # Search and Filter Options
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("Search in feedback", "")
        with col2:
            selected_response = st.multiselect(
                "Filter by Response Type",
                options=["Always", "Usually", "Sometimes", "Never"],
                default=[]
            )
        with col3:
            date_sort = st.radio("Sort by", ["Newest", "Oldest"])

        # Apply filters
        display_df = filtered_df.copy()
        if search_term:
            mask = display_df.apply(lambda x: x.astype(str).str.contains(search_term, case=False)).any(axis=1)
            display_df = display_df[mask]
        
        if selected_response:
            response_mask = display_df['call_button_response'].str.contains('|'.join(selected_response), case=False)
            display_df = display_df[response_mask]

        # Sort by date
        display_df = display_df.sort_values('timestamp', 
                                          ascending=date_sort == "Oldest")

        # Show filtered feedback with configured columns
        if not display_df.empty:
            st.dataframe(
                display_df[[
                    'timestamp', 'room_number', 'stay_feedback',
                    'call_button_response', 'bathroom_help_frequency',
                    'nurse_explanation_clarity', 'nurse_listening',
                    'nurse_courtesy', 'recognition'
                ]].sort_values('timestamp', ascending=False),
                column_config={
                    'stay_feedback': st.column_config.TextColumn('Feedback', width='medium'),
                    'recognition': st.column_config.TextColumn('Recognition', width='medium'),
                    'timestamp': st.column_config.DatetimeColumn('Date/Time', format='MMM DD, YYYY HH:mm')
                },
                hide_index=True
            )
            
            # Show record count
            st.info(f"Showing {len(display_df)} of {len(filtered_df)} records")
        else:
            st.warning("No records match the current filters.")

        # Add download buttons for detailed feedback
        st.subheader("Download Options")
        download_col1, download_col2, download_col3 = st.columns(3)
        
        with download_col1:
            # CSV download
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv,
                file_name=f"feedback_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download the filtered feedback data as CSV"
            )
        
        with download_col2:
            # Excel download
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, sheet_name='Feedback Data', index=False)
                
                # Add insights to a new sheet
                insights_data = []
                for metric, stats in metrics_analysis.items():
                    insights_data.extend([
                        [metric, 'Always', f"{stats['always_pct']:.1f}%"],
                        [metric, 'Usually', f"{stats['usually_pct']:.1f}%"],
                        [metric, 'Sometimes', f"{stats['sometimes_pct']:.1f}%"],
                        [metric, 'Never', f"{stats['never_pct']:.1f}%"],
                        ['', '', '']  # Empty row for spacing
                    ])
                
                insights_df = pd.DataFrame(insights_data, columns=['Metric', 'Response', 'Percentage'])
                insights_df.to_excel(writer, sheet_name='Insights', index=False)
                
            st.download_button(
                label="📥 Download Report (Excel)",
                data=buffer.getvalue(),
                file_name=f"feedback_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download the feedback data and insights as Excel"
            )
        
        with download_col3:
            # Generate PDF-ready summary
            summary_text = f"""Hospital Stay Feedback Report
Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}

Key Metrics Summary:
• Total Responses: {total_responses}
• Overall Excellence Rate: {overall_always:.1f}%
• Most Consistent Area: {max(metrics_analysis.items(), key=lambda x: x[1]['always_pct'])[0]}
• Primary Focus Area: {max(metrics_analysis.items(), key=lambda x: x[1]['sometimes_pct'] + x[1]['never_pct'])[0]}

Detailed Metrics:
"""
            # Add detailed metrics
            for metric, stats in metrics_analysis.items():
                summary_text += f"\n{metric}:\n"
                summary_text += f"• Always: {stats['always_pct']:.1f}%\n"
                summary_text += f"• Usually: {stats['usually_pct']:.1f}%\n"
                summary_text += f"• Sometimes/Never: {stats['sometimes_pct'] + stats['never_pct']:.1f}%\n"
            
            # Add download button for summary
            st.download_button(
                label="📥 Download Summary (TXT)",
                data=summary_text,
                file_name=f"feedback_summary_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.txt",
                mime="text/plain",
                help="Download a text summary of the feedback analysis"
            )

    except Exception as e:
        logger.error(f"Error in reports page: {e}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

# Define the navigation structure
PAGES = {
    "Patient Feedback Form": "web_form",
    "Feedback Reports": "reports"
}

# Update the main function with the new navigation
def main():
    # Create a single database connection for the session
    try:
        # Use absolute path for database file
        db_path = os.path.join(project_root, "feedback.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Initialize database if needed
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
        if not cursor.fetchone():
            init_db(conn)
            logger.info("Database initialized successfully")
        
        # Title of the main page
        st.title("Hospital Stay Feedback")

        # Create navigation using tabs
        tab1, tab2 = st.tabs(["Patient Feedback Form", "Feedback Reports"])

        # Add container for scrollable content
        with tab1:
            web_form_page(conn)
        
        with tab2:
            reports_page(conn)

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        st.error("A database error occurred. Please try again later.")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()