import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
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

# Now we can import from src
from src.sample_data import get_sample_feedback

# Set page config for full-width layout
st.set_page_config(layout="wide")

# Database setup
DB_FILE = "feedback.db"

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

def plot_response_distribution(df, column, title):
    """Create a bar chart showing response distribution."""
    if df.empty:
        return None
    
    # Group by the specified column and count the occurrences
    response_counts = df.groupby(column).size().reset_index(name='counts')
    
    # Create an Altair chart for the response distribution
    chart = alt.Chart(response_counts).mark_bar().encode(
        x=alt.X(f'{column}:N', title=title),  # Use 'N' for nominal data
        y=alt.Y('counts:Q', title='Count'),   # Use 'Q' for quantitative data
        color=alt.Color(f'{column}:N', legend=None)  # Optional: remove legend if not needed
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

def init_db(conn):
    # Drop the existing feedback table if it exists (WARNING: This will delete all existing data)
    conn.execute("DROP TABLE IF EXISTS feedback")
    
    # Create a new feedback table with the timestamp column
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

    # Insert new sample data
    sample_feedback = get_sample_feedback()
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO feedback (
            room_number, stay_feedback, call_button_response,
            bathroom_help_frequency, nurse_explanation_clarity,
            nurse_listening, nurse_courtesy, recognition, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_feedback)
    conn.commit()
    logging.info(f"Inserted {len(sample_feedback)} sample feedback entries.")


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

# Define a function to process text for word cloud with sentiment analysis
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


# Define the navigation structure
PAGES = {
    "Web Form": "web_form",
    "Reports": "reports"
}

def main():
    # Database connection
    conn = sqlite3.connect(DB_FILE)
    init_db(conn)

    # Title of the main page
    st.title("Hospital Stay Feedback")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Page dispatcher
    if selection == "Web Form":
        web_form_page(conn)
    elif selection == "Reports":
        reports_page(conn)

def web_form_page(conn):
    st.subheader("Patient Feedback Form")

    # Create a simple form
    with st.form(key='feedback_form'):
        room_number = st.text_input(label="Room number", placeholder="Enter your room number here", max_chars=3)
        stay_feedback = st.text_area(label="1. How has your stay been so far?", placeholder="Your feedback...")
        call_button_response = st.selectbox(
            "2. So far during your hospital stay, after you pressed your call button, how often did you get help as soon as you wanted?",
            options=["Always", "Usually", "Sometimes", "Never"]
        )
        bathroom_help_frequency = st.selectbox(
            "3. How often do you get help in getting to the bathroom or in using the bedpan as soon as you wanted?",
            options=["Always", "Usually", "Sometimes", "Never"]
        )
        nurse_explanation_clarity = st.selectbox(
            "4. So far during your hospital stay, how often are the nurses explaining things in a way you can understand?",
            options=["Always", "Usually", "Sometimes", "Never"]
        )
        nurse_listening = st.selectbox(
            "5. So far during your hospital stay, how often did nurses listen carefully to you?",
            options=["Always", "Usually", "Sometimes", "Never"]
        )
        nurse_courtesy = st.selectbox(
            "6. So far during your hospital stay, how often are nurses treating you with courtesy and respect?",
            options=["Always", "Usually", "Sometimes", "Never"]
        )
        recognition = st.text_area(
            label="7. Is there anyone you would like to recognize for doing an excellent job? (Don't forget Daisy, Shine, and MOC nominations if needed.)",
            placeholder="Recognition or nomination..."
        )
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            # Input validation for room number
            if room_number.isdigit() and len(room_number) == 3:
                # Save the feedback to the database with timestamp
                insert_feedback(conn, (
                    room_number, stay_feedback, call_button_response,
                    bathroom_help_frequency, nurse_explanation_clarity,
                    nurse_listening, nurse_courtesy, recognition,
                    datetime.now().isoformat()
                ))
                st.success("Thank you for your feedback!")
            else:
                st.error("Room number must be a 3-digit number.")
        

def reports_page(conn):
    st.subheader("Reports")

    try:
        # Fetch and process data
        df = pd.read_sql_query("SELECT * FROM feedback", conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if df.empty:
            st.warning("No feedback data available.")
            return
        
        # Sidebar filters
        st.sidebar.title("Filters")
        
        # Date range filter
        st.sidebar.subheader("Date Range")
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Convert date inputs to datetime for filtering
        start_datetime = pd.Timestamp(start_date)
        end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        
        # Text search filter
        search_text = st.sidebar.text_input("Search in feedback or recognition", "")
        
        # Existing filters
        room_filter = st.sidebar.multiselect("Filter by Room Number:", options=sorted(df['room_number'].unique()))
        call_button_filter = st.sidebar.multiselect("Filter by Call Button Response:", options=sorted(df['call_button_response'].unique()))
        bathroom_help_filter = st.sidebar.multiselect("Filter by Bathroom Help Frequency:", options=sorted(df['bathroom_help_frequency'].unique()))
        nurse_explanation_filter = st.sidebar.multiselect("Filter by Nurse Explanation Clarity:", options=sorted(df['nurse_explanation_clarity'].unique()))
        nurse_listening_filter = st.sidebar.multiselect("Filter by Nurse Listening:", options=sorted(df['nurse_listening'].unique()))
        nurse_courtesy_filter = st.sidebar.multiselect("Filter by Nurse Courtesy:", options=sorted(df['nurse_courtesy'].unique()))
        
        # Sort options
        sort_column = st.sidebar.selectbox(
            "Sort by:",
            options=['timestamp', 'room_number', 'call_button_response', 'nurse_courtesy'],
            index=0
        )
        sort_order = st.sidebar.radio("Sort order:", ["Descending", "Ascending"])
        
        # Apply filters to the DataFrame
        mask = (df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)
        
        if search_text:
            text_mask = (
                df['stay_feedback'].str.contains(search_text, case=False, na=False) |
                df['recognition'].str.contains(search_text, case=False, na=False)
            )
            mask = mask & text_mask
        
        if room_filter:
            mask = mask & df['room_number'].isin(room_filter)
        if call_button_filter:
            mask = mask & df['call_button_response'].isin(call_button_filter)
        if bathroom_help_filter:
            mask = mask & df['bathroom_help_frequency'].isin(bathroom_help_filter)
        if nurse_explanation_filter:
            mask = mask & df['nurse_explanation_clarity'].isin(nurse_explanation_filter)
        if nurse_listening_filter:
            mask = mask & df['nurse_listening'].isin(nurse_listening_filter)
        if nurse_courtesy_filter:
            mask = mask & df['nurse_courtesy'].isin(nurse_courtesy_filter)
        
        filtered_df = df[mask]
        
        # Sort the DataFrame
        filtered_df = filtered_df.sort_values(
            by=sort_column,
            ascending=(sort_order == "Ascending")
        )
        
        # Data quality warnings
        if len(filtered_df) < 5:
            st.warning("⚠️ Very few entries match your current filters. Results may not be statistically significant.")
        
        # Enhanced Summary Metrics with Trends
        st.subheader("Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_responses = len(filtered_df)
            prev_period = filtered_df['timestamp'] < (pd.Timestamp.now() - timedelta(days=7))
            prev_total = len(filtered_df[prev_period])
            delta = ((total_responses - prev_total) / prev_total * 100) if prev_total > 0 else 0
            st.metric("Total Responses", total_responses, f"{delta:.1f}%")
        
        with col2:
            always_courtesy = len(filtered_df[filtered_df['nurse_courtesy'] == 'Always'])
            courtesy_pct = (always_courtesy / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            st.metric("Always Courteous", f"{courtesy_pct:.1f}%")
        with col3:
            quick_response = len(filtered_df[filtered_df['call_button_response'] == 'Always'])
            response_pct = (quick_response / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            st.metric("Always Quick Response", f"{response_pct:.1f}%")
        with col4:
            recognition_count = len(filtered_df[filtered_df['recognition'].str.contains(r'[A-Za-z]', na=False)])
            st.metric("Staff Recognitions", recognition_count)
        
        # Response Distribution Visualization
        st.subheader("Response Distribution")
        metric_options = [
            "call_button_response",
            "bathroom_help_frequency",
            "nurse_explanation_clarity",
            "nurse_listening",
            "nurse_courtesy"
        ]
        selected_metric = st.selectbox("Select metric to visualize:", metric_options)
        
        chart = plot_response_distribution(filtered_df, selected_metric, f"Distribution of {selected_metric}")
        if chart:
            st.altair_chart(chart, use_container_width=True)
        
        # Trend Analysis
        st.subheader("Trend Analysis")
        trend_metric = st.selectbox("Select metric for trend analysis:", metric_options)
        trend_window = st.slider("Rolling average window (days)", 3, 30, 7)
        
        trend_chart = plot_trend_analysis(filtered_df, trend_metric, trend_window)
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True)
        
        # Interactive Word Cloud
        st.subheader("Feedback Analysis")
        
        # Category filter for word cloud
        category_options = ['All Categories', 'Responsiveness', 'Care Quality', 'Comfort', 'Staff Behavior']
        selected_category = st.selectbox("Focus on category:", category_options)
        
        # Generate word cloud with selected category
        text = " ".join(feedback for feedback in filtered_df['stay_feedback'])
        if text.strip():
            wordcloud, categories, sentiments = generate_sentiment_wordcloud(
                text,
                filtered_df,
                category=selected_category.lower() if selected_category != 'All Categories' else None,
                start_date=start_date,
                end_date=end_date
            )
            
            # Display traditional word cloud
            st.subheader("Word Cloud Visualization")
            fig_wordcloud, ax_wordcloud = plt.subplots(figsize=(15, 8))
            ax_wordcloud.imshow(wordcloud, interpolation='bilinear')
            ax_wordcloud.axis('off')
            st.pyplot(fig_wordcloud)
            
            # Download button for word cloud
            wordcloud_png = fig_to_png(fig_wordcloud)
            st.download_button(
                "Download Word Cloud as PNG",
                wordcloud_png,
                "wordcloud.png",
                "image/png"
            )
            
            # Interactive word analysis
            st.subheader("Interactive Word Analysis")
            words_df = pd.DataFrame({
                'word': list(sentiments.keys()),
                'sentiment': list(sentiments.values()),
                'category': [categories.get(word, 'other') for word in sentiments.keys()]
            })
            
            fig_scatter = px.scatter(
                words_df,
                x='sentiment',
                y='category',
                size=[abs(s) * 20 for s in words_df['sentiment']],
                color='sentiment',
                hover_data=['word'],
                color_continuous_scale='RdYlGn',
                title='Word Sentiment Analysis'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Download button for scatter plot
            scatter_png = fig_to_png(fig_scatter)
            st.download_button(
                "Download Sentiment Analysis as PNG",
                scatter_png,
                "sentiment_analysis.png",
                "image/png",
                key='download-scatter'
            )
            
            # Show related feedback for selected word
            selected_word = st.selectbox("Select word to see related feedback:", words_df['word'].tolist())
            if selected_word:
                related_feedback = filtered_df[
                    filtered_df['stay_feedback'].str.contains(selected_word, case=False, na=False)
                ]
                st.write(f"Feedback containing '{selected_word}':")
                st.dataframe(
                    related_feedback[['timestamp', 'stay_feedback']],
                    use_container_width=True
                )
                
                # Download button for related feedback
                related_csv = related_feedback.to_csv(index=False).encode('utf-8')
                st.download_button(
                    f"Download '{selected_word}' feedback as CSV",
                    related_csv,
                    f"feedback_{selected_word}.csv",
                    "text/csv",
                    key='download-related'
                )
        
        # Display the filtered entries
        st.subheader("Detailed Feedback")
        
        # Add table search functionality
        table_search = st.text_input(
            "Search in table (multiple terms separated by space):",
            placeholder="Enter search terms...",
            help="Shows only rows that contain ALL entered terms (case-insensitive)"
        )
        
        # Option to show/hide columns
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to display:",
            all_columns,
            default=['timestamp', 'room_number', 'stay_feedback', 'recognition']
        )
        
        # Apply table search filter if search terms exist
        display_df = filtered_df.copy()
        if table_search.strip():
            search_terms = table_search.lower().split()
            mask = pd.Series(True, index=display_df.index)
            
            for term in search_terms:
                term_mask = pd.Series(False, index=display_df.index)
                for column in selected_columns:
                    term_mask |= display_df[column].astype(str).str.lower().str.contains(term, na=False)
                mask &= term_mask
            
            display_df = display_df[mask]
        
        # Display the filtered DataFrame with selected columns
        if not display_df.empty:
            st.dataframe(display_df[selected_columns], use_container_width=True)
            st.caption(f"Showing {len(display_df)} of {len(filtered_df)} entries")
        else:
            st.warning("No data matches the current filters and search terms.")
        
        # Export option
        if not display_df.empty:
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download filtered data as CSV",
                csv,
                "hospital_feedback.csv",
                "text/csv",
                key='download-csv'
            )
    
    except Exception as e:
        logging.error(f"Error in reports page: {e}")
        st.error("An error occurred while generating the report. Please try again later.")

# Add these utility functions at the top of the file with other imports
def fig_to_png(fig):
    """Convert a matplotlib or plotly figure to PNG bytes."""
    if isinstance(fig, plt.Figure):
        # For matplotlib figures
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        return buf.getvalue()
    else:
        # For plotly figures
        return fig.to_image(format='png')

def chart_to_png(chart):
    """Convert an Altair chart to PNG bytes."""
    return chart.save(format='png').getvalue()

if __name__ == "__main__":
    main()
