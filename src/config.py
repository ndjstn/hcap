import os
from dataclasses import dataclass, field
from typing import List, ClassVar

@dataclass
class Config:
    DB_FILE: ClassVar[str] = "feedback.db"
    LOG_FORMAT: ClassVar[str] = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_LEVEL: ClassVar[str] = "INFO"
    
    NLTK_DATA_PATH: ClassVar[str] = os.getenv("NLTK_DATA_PATH", "nltk_data")
    SPACY_MODEL: ClassVar[str] = "en_core_web_sm"
    
    WORDCLOUD_WIDTH: ClassVar[int] = 1200
    WORDCLOUD_HEIGHT: ClassVar[int] = 600
    
    FREQUENCY_OPTIONS: ClassVar[List[str]] = ["Always", "Usually", "Sometimes", "Never"]