import os
from dataclasses import dataclass

@dataclass
class Config:
    DB_FILE: str = "feedback.db"
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_LEVEL: str = "INFO"
    
    # Add any other configuration settings here
    NLTK_DATA_PATH: str = os.getenv("NLTK_DATA_PATH", "nltk_data")
    SPACY_MODEL: str = "en_core_web_sm"
    
    # Visualization settings
    WORDCLOUD_WIDTH: int = 1200
    WORDCLOUD_HEIGHT: int = 600
    
    # Form options
    FREQUENCY_OPTIONS: list = ["Always", "Usually", "Sometimes", "Never"] 