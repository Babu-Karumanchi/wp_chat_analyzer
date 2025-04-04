import re
import pandas as pd
import emoji
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_chat(file_upload):
    """Read and preprocess WhatsApp chat export file."""
    
    # Check file type
    if file_upload is None:
        return None
    
    # Read the file content
    try:
        content = file_upload.getvalue().decode('utf-8')
    except UnicodeDecodeError:
        try:
            content = file_upload.getvalue().decode('utf-8-sig')
        except UnicodeDecodeError:
            try:
                content = file_upload.getvalue().decode('latin-1')
            except:
                return None
    
    # Extract messages with regex pattern
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s[APap][Mm])\s-\s(.*?):\s(.*?)(?=\n\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s[APap][Mm]\s-\s|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        # Try alternative pattern without AM/PM
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?)\s-\s(.*?):\s(.*?)(?=\n\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?)\s-\s|$)'
        matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        return None
    
    # Convert to dataframe
    df = pd.DataFrame(matches, columns=['timestamp', 'user', 'message'])
    
    # Clean timestamp and convert to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    
    # Extract media messages
    df['has_media'] = df['message'].str.contains('<Media omitted>|image omitted|video omitted|sticker omitted', case=False)
    
    # Extract URLs
    url_pattern = r'(https?://[^\s]+)'
    df['has_url'] = df['message'].str.contains(url_pattern)
    
    # Extract emojis
    df['emojis'] = df['message'].apply(lambda text: [c for c in text if c in emoji.EMOJI_DATA])
    df['emoji_count'] = df['emojis'].str.len()
    
    # Extract message length
    df['message_length'] = df['message'].str.len()
    
    # Add date columns for easier analysis
    df['date'] = df['timestamp'].dt.date
    df['day'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month_name()
    df['year'] = df['timestamp'].dt.year
    
    # Clean user names (remove trailing/leading whitespace)
    df['user'] = df['user'].str.strip()
    
    return df

def identify_group_members(df):
    """Return a list of unique users in the chat."""
    if df is None or 'user' not in df.columns:
        return []
    return sorted(df['user'].unique())

def extract_common_words(df, num_words=20, exclude_users=None):
    """Extract the most common words in the chat, excluding stopwords."""
    
    if df is None or 'message' not in df.columns:
        return {}
    
    # Filter by users if specified
    if exclude_users:
        df = df[~df['user'].isin(exclude_users)]
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Combine all messages
    all_messages = ' '.join(df['message'].tolist())
    
    # Tokenize
    words = re.findall(r'\b[a-zA-Z]{3,15}\b', all_messages.lower())
    
    # Remove stopwords and filter by length
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count occurrences
    word_counts = Counter(filtered_words)
    
    # Return the most common words
    return dict(word_counts.most_common(num_words))

def identify_chat_type(df):
    """Determine if the chat is a group chat or individual chat."""
    if df is None or 'user' not in df.columns:
        return "Unknown"
    
    unique_users = df['user'].nunique()
    if unique_users > 2:
        return "Group Chat"
    else:
        return "Individual Chat"
