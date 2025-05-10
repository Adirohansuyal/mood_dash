import streamlit as st
import base64

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
from PIL import Image
import io
import uuid
import streamlit as st

# Inject CSS
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(to bottom, darkgreen, lightgreen); /* Gradient background */
    }}
    body {{
        font-family: 'Segoe UI', sans-serif;
        color: orange;
    }}
    .main {{
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
        color: orange;
    }}
    h1 {{
        color: darkorange;
        font-weight: 600;
    }}
    h2 {{
        color: orange;
        font-weight: 600;
    }}
    .stButton button {{
        border-radius: 8px;
        background-color: #007BFF;
        color: white;
        padding: 0.5rem 1.5rem;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }}
    .stButton button:hover {{
        background-color: #0056b3;
    }}
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
    
    # Calculate average emotion scores
    avg_scores = filtered_df.groupby('emotion')['emotion_score'].mean().reset_index()
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(filtered_df))
    with col2:
        st.metric("Most Frequent Mood", f"{most_frequent} {get_emotion_emoji(most_frequent)}")
    with col3:
        st.metric("Unique Sessions", filtered_df['session_id'].nunique())
    
    # Create visualizations
    st.subheader("Mood Distribution")
    
    # Plot emotion distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # Reduced size
    
    # Pie chart for emotion distribution
    emotion_counts = filtered_df['emotion'].value_counts()
    ax1.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', 
            shadow=True, startangle=90)
    ax1.axis('equal')
    ax1.set_title('Emotion Distribution')
    
    # Line chart for emotion over time
    time_data = filtered_df.groupby(filtered_df['timestamp'].dt.date)['emotion'].value_counts().unstack().fillna(0)
    time_data.plot(kind='line', ax=ax2, marker='o')
    ax2.set_title('Emotions Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Count')
    ax2.legend(title='Emotion')
    
    st.pyplot(fig)
    
    # Intensity heatmap
    st.subheader("Emotion Intensity Heatmap")
    
    # Group by date and emotion, averaging the scores
    heatmap_data = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'emotion'])['emotion_score'].mean().unstack().fillna(0)
    
    fig, ax = plt.subplots(figsize=(8, 4))  # Reduced size
    sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax, annot=True, fmt=".2f")
    ax.set_title('Emotion Intensity Over Time')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Date')
    
    st.pyplot(fig)
    
    # Show raw data
    st.subheader("Raw Data")
    st.dataframe(filtered_df[['timestamp', 'emotion', 'emotion_score']])
    
    # Option to download data
    csv = filtered_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="mood_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
    # Allover Mood Analysis
    st.header("📈 Allover Mood Analysis (Historical)")

    overall_df = st.session_state.mood_data.copy()
    overall_df['timestamp'] = pd.to_datetime(overall_df['timestamp'])

    if not overall_df.empty:
        # Bar chart for total mood counts
        st.subheader("Overall Mood Count")
        mood_counts = overall_df['emotion'].value_counts().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))  # Reduced size
        sns.barplot(x=mood_counts.index, y=mood_counts.values, ax=ax, palette="Set2")
        ax.set_title("Total Occurrences of Each Mood")
        ax.set_ylabel("Count")
        ax.set_xlabel("Emotion")

        for i, v in enumerate(mood_counts.values):
            ax.text(i, v + 0.5, str(int(v)), ha='center', va='bottom')

        st.pyplot(fig)

        # Average emotion score per mood
        st.subheader("Average Emotion Score per Mood")
        avg_overall_scores = overall_df.groupby('emotion')['emotion_score'].mean().reset_index()

        fig2, ax2 = plt.subplots(figsize=(8, 4))  # Reduced size
        sns.barplot(data=avg_overall_scores, x='emotion', y='emotion_score', ax=ax2, palette="coolwarm")
        ax2.set_title("Average Emotion Intensity (All Time)")
        ax2.set_ylabel("Average Score")
        ax2.set_xlabel("Emotion")

        for i, row in avg_overall_scores.iterrows():
            ax2.text(i, row['emotion_score'] + 0.5, f"{row['emotion_score']:.2f}", ha='center', va='bottom')

        st.pyplot(fig2)

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
        
        with col2:
            # Tips for better mood analysis
            st.subheader("Tips for Better Analysis")
            st.markdown("""
            - Ensure your face is clearly visible
            - Good lighting improves accuracy
            - Face the camera directly
            - Try to express natural emotions
            - Remove glasses if possible
            """)
            
            # Show a brief summary of recent moods
            if not st.session_state.mood_data.empty:
                st.subheader("Your Recent Moods")
                recent_moods = st.session_state.mood_data.tail(5).copy()
                recent_moods['timestamp'] = pd.to_datetime(recent_moods['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                recent_moods['emoji'] = recent_moods['emotion'].apply(get_emotion_emoji)
                recent_moods['display'] = recent_moods['emotion'] + ' ' + recent_moods['emoji']
                st.dataframe(recent_moods[['timestamp', 'display', 'emotion_score']], hide_index=True)
    
    # Tab 2: Mood History
    with tab2:
        show_mood_analysis()
    
    # Tab 3: Settings
    with tab3:
        st.header("⚙️ Settings")
        
        # Session management
        st.subheader("Session Management")
        if st.button("Start New Session"):
            st.session_state.current_session_id = str(uuid.uuid4())
            st.success(f"New session started: {st.session_state.current_session_id}")
        
        st.info(f"Current session ID: {st.session_state.current_session_id}")
        
        # Data management
        st.subheader("Data Management")
        st.warning("Warning: These actions cannot be undone!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Session Data"):
                session_mask = st.session_state.mood_data['session_id'] != st.session_state.current_session_id
                st.session_state.mood_data = st.session_state.mood_data[session_mask]
                save_mood_data()
                st.success("Current session data cleared!")
                
        with col2:
            if st.button("Clear All Data"):
                st.session_state.mood_data = pd.DataFrame(columns=['timestamp', 'emotion', 'emotion_score', 'session_id'])
                save_mood_data()
                st.success("All data cleared!")

if __name__ == "__main__":
    main()
