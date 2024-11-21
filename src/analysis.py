import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timedelta
import logging
from src.config import Config
from src.logger import get_logger
import numpy as np
from typing import Dict, List, Tuple

logger = get_logger(__name__)

def setup_nlp():
    """Initialize NLP components."""
    try:
        logger.info("Setting up NLP components")
        
        # Download NLTK data
        nltk.download('vader_lexicon', quiet=True)
        
        # Load SpaCy model
        try:
            nlp = spacy.load(Config.SPACY_MODEL)
        except OSError:
            logger.info(f"Downloading SpaCy model: {Config.SPACY_MODEL}")
            from spacy.cli import download
            download(Config.SPACY_MODEL)
            nlp = spacy.load(Config.SPACY_MODEL)
        
        logger.info("Successfully initialized NLP components")
        return nlp, SentimentIntensityAnalyzer()
    except Exception as e:
        logger.error(f"Failed to setup NLP components: {e}")
        raise

def analyze_feedback(df, nlp, sentiment_analyzer, start_date=None, end_date=None):
    """Analyze feedback data for insights."""
    try:
        if df.empty:
            logger.warning("No feedback data to analyze")
            return {}
        
        logger.info("Starting feedback analysis")
        
        # Filter by date range if provided
        if start_date and end_date:
            try:
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                
                mask = (df['timestamp'].dt.date >= start_ts.date()) & \
                       (df['timestamp'].dt.date <= end_ts.date())
                df = df[mask]
                logger.info(f"Filtered data to date range: {start_date} to {end_date}")
            except Exception as e:
                logger.error(f"Error filtering by date range: {e}")
                raise
        
        results = {
            'total_responses': len(df),
            'sentiment_scores': calculate_sentiment_scores(df, sentiment_analyzer),
            'response_metrics': calculate_response_metrics(df),
            'common_themes': extract_common_themes(df, nlp)
        }
        
        logger.info("Successfully completed feedback analysis")
        return results
        
    except Exception as e:
        logger.error(f"Error during feedback analysis: {e}")
        raise

def calculate_sentiment_scores(df, sentiment_analyzer):
    """Calculate sentiment scores for feedback text."""
    scores = []
    for text in df['stay_feedback']:
        if isinstance(text, str):
            score = sentiment_analyzer.polarity_scores(text)
            scores.append(score)
    return pd.DataFrame(scores).mean().to_dict()

def calculate_response_metrics(df):
    """Calculate metrics for different response categories."""
    metrics = {}
    for column in ['call_button_response', 'nurse_courtesy', 'nurse_listening']:
        if column in df.columns:
            metrics[column] = df[column].value_counts().to_dict()
    return metrics

def extract_common_themes(df, nlp):
    """Extract common themes from feedback text using NLP."""
    themes = []
    for text in df['stay_feedback']:
        if isinstance(text, str):
            doc = nlp(text)
            # Extract relevant noun phrases and entities
            themes.extend([chunk.text.lower() for chunk in doc.noun_chunks])
    
    return pd.Series(themes).value_counts().head(10).to_dict()

def calculate_summary_metrics(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Calculate summary metrics with period-over-period comparison.
    
    Args:
        current_df: DataFrame for current period
        previous_df: DataFrame for previous period
    Returns:
        Dict containing metrics and their changes
    """
    metrics = {}
    
    # Total responses
    metrics['total_responses'] = {
        'current': len(current_df),
        'previous': len(previous_df),
        'change': calculate_percentage_change(len(current_df), len(previous_df))
    }
    
    # Always courteous percentage
    current_courtesy = calculate_always_percentage(current_df, 'nurse_courtesy')
    prev_courtesy = calculate_always_percentage(previous_df, 'nurse_courtesy')
    metrics['courtesy'] = {
        'current': current_courtesy,
        'previous': prev_courtesy,
        'change': current_courtesy - prev_courtesy
    }
    
    # Quick response percentage
    current_response = calculate_always_percentage(current_df, 'call_button_response')
    prev_response = calculate_always_percentage(previous_df, 'call_button_response')
    metrics['response'] = {
        'current': current_response,
        'previous': prev_response,
        'change': current_response - prev_response
    }
    
    return metrics

def calculate_percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change between two values."""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def calculate_always_percentage(df: pd.DataFrame, column: str) -> float:
    """Calculate percentage of 'Always' responses for a given column."""
    if len(df) == 0:
        return 0
    always_count = len(df[df[column].str.startswith('Always')])
    return (always_count / len(df)) * 100

def analyze_trends(
    df: pd.DataFrame,
    metrics: List[str],
    window: int = 7
) -> Dict[str, pd.Series]:
    """
    Analyze trends for specified metrics.
    
    Args:
        df: DataFrame containing feedback data
        metrics: List of metric columns to analyze
        window: Rolling window size
    Returns:
        Dict containing trend analysis for each metric
    """
    trends = {}
    response_scores = {'Always': 4, 'Usually': 3, 'Sometimes': 2, 'Never': 1}
    
    for metric in metrics:
        daily_scores = df.groupby(df['timestamp'].dt.date)[metric].agg(
            lambda x: x.map(response_scores).mean()
        )
        trends[metric] = {
            'raw': daily_scores,
            'rolling_avg': daily_scores.rolling(window).mean(),
            'trend_direction': calculate_trend_direction(daily_scores)
        }
    
    return trends

def calculate_trend_direction(series: pd.Series) -> str:
    """Calculate the trend direction based on recent data."""
    if len(series) < 2:
        return "neutral"
    
    recent_change = series.iloc[-1] - series.iloc[-2]
    if recent_change > 0:
        return "improving"
    elif recent_change < 0:
        return "declining"
    return "stable" 