import sqlite3
import logging
from datetime import datetime
import pandas as pd
from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)

def get_database_connection():
    """Create and return a database connection."""
    try:
        conn = sqlite3.connect(Config.DB_FILE)
        logger.info("Successfully established database connection")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def init_db(conn):
    """Initialize the database only if it doesn't exist or is empty."""
    try:
        cursor = conn.cursor()
        
        # Create feedback table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        
        # Check if table is empty before adding sample data
        cursor.execute("SELECT COUNT(*) FROM feedback")
        if cursor.fetchone()[0] == 0:
            from sample_data import get_sample_feedback
            sample_feedback = get_sample_feedback()
            cursor.executemany("""
                INSERT INTO feedback (
                    room_number, stay_feedback, call_button_response,
                    bathroom_help_frequency, nurse_explanation_clarity,
                    nurse_listening, nurse_courtesy, recognition, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, sample_feedback)
            conn.commit()
            logger.info(f"Initialized database with {len(sample_feedback)} sample records")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise

def insert_feedback(conn, feedback_data):
    """Insert new feedback into the database."""
    try:
        cursor = conn.cursor()
        
        # First check if feedback_data is None or empty
        if not feedback_data:
            error_msg = (
                "No feedback data provided. "
                "Please ensure you've filled out the form."
            )
            logger.error(error_msg)
            return False, error_msg
        
        # Validate input data length with detailed error
        if len(feedback_data) != 9:
            error_msg = (
                f"Form data is incomplete: Expected 9 fields, got {len(feedback_data)}. "
                "Please fill out all required sections of the form."
            )
            logger.error(error_msg)
            return False, error_msg
        
        # Validate data types and content
        try:
            room_number = str(feedback_data[0]).strip()
            stay_feedback = str(feedback_data[1]).strip() if feedback_data[1] else ''
            call_button = str(feedback_data[2]).strip()
            bathroom_help = str(feedback_data[3]).strip()
            nurse_explanation = str(feedback_data[4]).strip()
            nurse_listening = str(feedback_data[5]).strip()
            nurse_courtesy = str(feedback_data[6]).strip()
            recognition = str(feedback_data[7]).strip() if feedback_data[7] else ''
            timestamp = str(feedback_data[8]).strip()
        except (ValueError, AttributeError) as e:
            error_msg = (
                "Invalid data format detected. "
                f"Error in field: {str(e)}. "
                "Please check your input and try again."
            )
            logger.error(f"{error_msg} Raw data: {feedback_data}")
            return False, error_msg
        
        # Validate required fields
        required_fields = {
            'Room Number': room_number,
            'Call Button Response': call_button,
            'Bathroom Help': bathroom_help,
            'Nurse Explanations': nurse_explanation,
            'Nurse Listening': nurse_listening,
            'Nurse Courtesy': nurse_courtesy
        }
        
        missing_fields = [field for field, value in required_fields.items() 
                         if not value or value.isspace()]
        
        if missing_fields:
            error_msg = (
                "Please complete the following required fields:\n"
                f"• {'\n• '.join(missing_fields)}"
            )
            logger.error(f"Missing required fields: {missing_fields}")
            return False, error_msg
        
        # Validate room number format
        if not room_number.isdigit() or len(room_number) != 3:
            error_msg = (
                "Invalid room number format. "
                "Please enter exactly 3 digits (e.g., 123)."
            )
            logger.error(f"Invalid room number: {room_number}")
            return False, error_msg
        
        # Validate response options
        valid_responses = {'Always', 'Usually', 'Sometimes', 'Never'}
        response_fields = {
            'Call Button Response': call_button,
            'Bathroom Help': bathroom_help,
            'Nurse Explanations': nurse_explanation,
            'Nurse Listening': nurse_listening,
            'Nurse Courtesy': nurse_courtesy
        }
        
        invalid_responses = [
            field for field, value in response_fields.items()
            if not any(resp in value for resp in valid_responses)
        ]
        
        if invalid_responses:
            error_msg = (
                "Invalid responses detected. Please select a valid option "
                "(Always, Usually, Sometimes, or Never) for:\n"
                f"• {'\n• '.join(invalid_responses)}"
            )
            logger.error(f"Invalid responses in fields: {invalid_responses}")
            return False, error_msg
        
        # Prepare sanitized data for insertion
        sanitized_data = (
            room_number,
            stay_feedback,
            call_button,
            bathroom_help,
            nurse_explanation,
            nurse_listening,
            nurse_courtesy,
            recognition,
            timestamp
        )
        
        # Log the sanitized data for debugging
        logger.debug(f"Attempting to insert data: {sanitized_data}")
        
        cursor.execute("""
            INSERT INTO feedback (
                room_number, stay_feedback, call_button_response,
                bathroom_help_frequency, nurse_explanation_clarity,
                nurse_listening, nurse_courtesy, recognition, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sanitized_data)
        
        conn.commit()
        inserted_id = cursor.lastrowid
        success_msg = f"Successfully saved feedback for room {room_number}"
        logger.info(f"{success_msg} (ID: {inserted_id})")
        return True, success_msg
        
    except sqlite3.IntegrityError as e:
        error_msg = (
            "Unable to save feedback: This appears to be a duplicate entry. "
            "Please ensure you haven't already submitted feedback for this room."
        )
        logger.error(f"{error_msg} Details: {str(e)}", exc_info=True)
        if conn:
            conn.rollback()
        return False, error_msg
        
    except sqlite3.Error as e:
        error_msg = (
            "Unable to save feedback to the database. "
            "Please try again in a few moments or contact support if the problem persists."
        )
        logger.error(f"{error_msg} Details: {str(e)}", exc_info=True)
        if conn:
            conn.rollback()
        return False, error_msg
        
    except Exception as e:
        error_msg = (
            "An unexpected error occurred while saving your feedback. "
            "Please try again or contact support if the problem persists.\n"
            f"Error details: {type(e).__name__}"
        )
        logger.error(f"{error_msg} Details: {str(e)}", exc_info=True)
        if conn:
            conn.rollback()
        return False, error_msg

def get_feedback_df(conn, filters=None):
    """Retrieve feedback data as a pandas DataFrame."""
    try:
        query = "SELECT * FROM feedback"
        df = pd.read_sql_query(query, conn)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        if filters:
            logger.debug(f"Applying filters: {filters}")
            for column, values in filters.items():
                if values:
                    df = df[df[column].isin(values)]
        
        logger.info(f"Retrieved {len(df)} feedback records")
        return df
    except Exception as e:
        logger.error(f"Error retrieving feedback: {e}")
        raise 