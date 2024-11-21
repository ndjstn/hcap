import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import logging
import altair as alt
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

def create_wordcloud(text, width=1200, height=600, max_words=50):
    """Create a basic wordcloud from text."""
    try:
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=max_words,
            min_font_size=10,
            max_font_size=80
        ).generate(text)
        return wordcloud
    except Exception as e:
        logger.error(f"Error creating wordcloud: {e}")
        return None

def save_wordcloud_image(wordcloud):
    """Save wordcloud to bytes buffer."""
    try:
        img_stream = io.BytesIO()
        wordcloud.to_image().save(img_stream, format='PNG')
        img_stream.seek(0)
        return img_stream
    except Exception as e:
        logger.error(f"Error saving wordcloud: {e}")
        return None

def plot_response_distribution(df: pd.DataFrame, selected_metrics: list = None) -> alt.Chart:
    """
    Create a comprehensive bar chart showing response distribution for selected metrics.
    
    Args:
        df: DataFrame containing feedback data
        selected_metrics: List of metric names to display (None means all metrics)
    """
    if df.empty:
        logger.warning("Empty dataframe provided to plot_response_distribution")
        return None
        
    metrics = {
        'Call Button': 'call_button_response',
        'Bathroom Help': 'bathroom_help_frequency',
        'Explanations': 'nurse_explanation_clarity',
        'Listening': 'nurse_listening',
        'Courtesy': 'nurse_courtesy'
    }
    
    # Filter metrics based on selection
    if selected_metrics:
        metrics = {k: v for k, v in metrics.items() if k in selected_metrics}
    
    # Prepare data for selected metrics
    plot_data = []
    for metric_name, column in metrics.items():
        if column in df.columns:
            try:
                counts = df[column].value_counts().reset_index()
                counts['Metric'] = metric_name
                counts.columns = ['Response', 'Count', 'Metric']
                plot_data.append(counts)
            except Exception as e:
                logger.warning(f"Error processing {metric_name}: {e}")
                continue
    
    if not plot_data:
        logger.warning("No valid data to plot in plot_response_distribution")
        return None
        
    plot_df = pd.concat(plot_data)
    
    try:
        # Create a single chart with grouped bars
        chart = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X('Response:N', sort=['Always', 'Usually', 'Sometimes', 'Never'],
                   axis=alt.Axis(title='Response Type')),
            y=alt.Y('Count:Q', title='Number of Responses'),
            color=alt.Color('Metric:N', 
                          legend=alt.Legend(
                              title='Metrics',
                              orient='top',
                              direction='horizontal'
                          )),
            xOffset='Metric:N',  # This creates the grouped bars
            tooltip=['Metric', 'Response', 'Count']
        ).properties(
            width=700,
            height=400,
            title=alt.TitleParams(
                text='Response Distribution Across Selected Metrics',
                fontSize=16
            )
        ).configure_axis(
            labelAngle=0,
            labelFontSize=12,
            titleFontSize=14
        ).configure_legend(
            labelFontSize=12,
            titleFontSize=14
        )
        
        return chart
    except Exception as e:
        logger.error(f"Error creating chart in plot_response_distribution: {e}")
        return None

def plot_trend_analysis(df: pd.DataFrame, selected_metrics: list = None, window: int = 7) -> go.Figure:
    """
    Create a comprehensive trend analysis showing selected metrics.
    
    Args:
        df: DataFrame containing feedback data
        selected_metrics: List of metric names to display (None means all metrics)
        window: Rolling average window size
    """
    if df.empty:
        logger.warning("Empty dataframe provided to plot_trend_analysis")
        return None
        
    metrics = {
        'Call Button': 'call_button_response',
        'Bathroom Help': 'bathroom_help_frequency',
        'Explanations': 'nurse_explanation_clarity',
        'Listening': 'nurse_listening',
        'Courtesy': 'nurse_courtesy'
    }
    
    # Filter metrics based on selection
    if selected_metrics:
        metrics = {k: v for k, v in metrics.items() if k in selected_metrics}
    
    fig = go.Figure()
    has_data = False
    
    for metric_name, column in metrics.items():
        if column in df.columns:
            try:
                # Convert responses to numeric scores
                response_scores = {'Always': 4, 'Usually': 3, 'Sometimes': 2, 'Never': 1}
                daily_scores = df.groupby(df['timestamp'].dt.date)[column].agg(
                    lambda x: x.map(response_scores).mean()
                ).rolling(window).mean()
                
                if not daily_scores.empty:
                    fig.add_trace(go.Scatter(
                        x=daily_scores.index,
                        y=daily_scores.values,
                        name=metric_name,
                        mode='lines+markers'
                    ))
                    has_data = True
            except Exception as e:
                logger.warning(f"Error processing {metric_name}: {e}")
                continue
    
    if not has_data:
        logger.warning("No valid data to plot in plot_trend_analysis")
        return None
    
    fig.update_layout(
        title=f'Trends Analysis for Selected Metrics ({window}-Day Rolling Average)',
        xaxis_title='Date',
        yaxis_title='Average Score (4=Always, 1=Never)',
        hovermode='x unified',
        showlegend=True,
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def generate_advanced_wordcloud(
    df: pd.DataFrame,
    text_column: str,
    start_date: Any = None,
    end_date: Any = None
) -> Tuple[WordCloud, Dict, Dict]:
    """
    Generate an advanced word cloud with sentiment analysis.
    
    Args:
        df: DataFrame containing feedback data
        text_column: Column name containing text data
        start_date: Start date for filtering
        end_date: End date for filtering
    Returns:
        Tuple containing WordCloud object, category dict, and sentiment dict
    """
    if start_date and end_date:
        mask = ((df['timestamp'].dt.date >= start_date) & 
                (df['timestamp'].dt.date <= end_date))
        text = ' '.join(df[mask][text_column].dropna())
    else:
        text = ' '.join(df[text_column].dropna())

    # Custom color function based on word frequency
    def color_func(word: str, font_size: int, position: Tuple, 
                  orientation: int, **kwargs) -> str:
        return f'hsl(230, 60%, {max(20, min(80, font_size/2))}%)'

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        color_func=color_func,
        max_words=100,
        min_font_size=10,
        max_font_size=80,
        prefer_horizontal=0.7,
        collocations=False
    ).generate(text)

    return wordcloud, {}, {}  # Placeholder for category and sentiment dicts 