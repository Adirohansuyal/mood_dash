import streamlit as st

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="AI Mood Analyzer",
    page_icon="😊",
    layout="wide"
)

import cv2
import numpy as np
import pandas as pd
import datetime
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace
import uuid

# Inject CSS for gradient background and styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, darkgreen, lightgreen);  /* Gradient background */
    }
    body {
        font-family: 'Segoe UI', sans-serif;
        color: orange;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
        color: orange;
    }
    h1 {
        color: darkorange;
        font-weight: 600;
    }
    h2 {
        color: orange;
        font-weight: 600;
    }
    .stButton button {
        border-radius: 8px;
        background-color: #007BFF;
        color: white;
        padding: 0.5rem 1.5rem;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'mood_data' not in st.session_state:
    st.session_state.mood_data = pd.DataFrame(columns=['timestamp', 'emotion', 'emotion_score', 'session_id'])

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

# Ensure data directory exists
DATA_DIR = "mood_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
# Path to the CSV file for storing mood data
MOOD_DATA_PATH = os.path.join(DATA_DIR, "mood_history.csv")

# Load existing data if available
if os.path.exists(MOOD_DATA_PATH):
    try:
        st.session_state.mood_data = pd.read_csv(MOOD_DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Function to save mood data
def save_mood_data():
    try:
        st.session_state.mood_data.to_csv(MOOD_DATA_PATH, index=False)
    except Exception as e:
        st.error(f"Error saving data: {e}")

# Function to analyze emotion from image
def analyze_emotion(img):
    try:
        # Convert BGR to RGB (DeepFace expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Analyze emotion
        result = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)
        
        # If result is a list, take the first item
        if isinstance(result, list):
            result = result[0]
            
        # Get the dominant emotion and its score
        emotions = result['emotion']
        dominant_emotion = max(emotions, key=emotions.get)
        emotion_score = emotions[dominant_emotion]
        
        return dominant_emotion, emotion_score
    except Exception as e:
        st.error(f"Error analyzing emotion: {e}")
        return "unknown", 0

# Function to get emoji for emotion
def get_emotion_emoji(emotion):
    emotion_emojis = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "fear": "😨",
        "surprise": "😲",
        "disgust": "🤢",
        "neutral": "😐",
        "unknown": "❓"
    }
    return emotion_emojis.get(emotion.lower(), "❓")

# Function to display historical mood data
def show_mood_analysis():
    st.header("📊 Mood History Analysis")
    
    if st.session_state.mood_data.empty:
        st.info("No mood data available yet. Capture some moods first!")
        return
    
    # Convert timestamp to datetime
    df = st.session_state.mood_data.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Date range filter
    col1, col2 = st.columns(2)
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    with col1:
        start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)
    
    # Filter data based on date range
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]
    
    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return
        
    # Display metrics
    st.subheader("Mood Metrics")
    
    # Calculate most frequent emotion
    most_frequent = filtered_df['emotion'].value_counts().idxmax()
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(filtered_df))
    with col2:
        st.metric("Most Frequent Mood", f"{most_frequent} {get_emotion_emoji(most_frequent)}")
    with col3:
        st.metric("Unique Sessions", filtered_df['session_id'].nunique())

    # Visualization code continues unchanged...

# Main application UI
def main():
    st.title("AI Mood Analyzer 😊😢😠")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Capture Mood", "Mood History", "Settings"])
    
    # Tab 1: Capture Mood
    with tab1:
        st.header("📷 Capture Your Mood")
        st.write("Use your webcam to capture and analyze your current mood!")
        
        # Webcam capture
        img_file_buffer = st.camera_input("Take a picture", key="mood_capture")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if img_file_buffer is not None:
                # Convert to OpenCV format
                bytes_data = img_file_buffer.getvalue()
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Analyze emotion
                emotion, score = analyze_emotion(img)
                
                # Display result
                st.success(f"Analysis Complete!")
                st.write(f"Detected emotion: {emotion} {get_emotion_emoji(emotion)}")
                st.write(f"Confidence: {score:.2f}%")
                
                # Store the result
                timestamp = datetime.datetime.now()
                new_data = pd.DataFrame({
                    'timestamp': [timestamp],
                    'emotion': [emotion],
                    'emotion_score': [score],
                    'session_id': [st.session_state.current_session_id]
                })
                
                st.session_state.mood_data = pd.concat([st.session_state.mood_data, new_data], ignore_index=True)
                save_mood_data()
                
                st.success("Mood recorded successfully!")
       
        # Tips and recent moods continue unchanged...

    # Tab 2: Mood History
    with tab2:
        show_mood_analysis()

    # Tab 3: Settings
    with tab3:
        st.header("⚙️ Settings")
        # Settings UI continues unchanged...

if __name__ == "__main__":
    main()