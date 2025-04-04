import pandas as pd
from collections import Counter
from textblob import TextBlob
import re

def get_basic_stats(df):
    """Calculate basic statistics about the chat."""
    if df is None or df.empty:
        return {}
    
    total_messages = len(df)
    total_users = df['user'].nunique()
    first_message_date = df['timestamp'].min().date()
    last_message_date = df['timestamp'].max().date()
    duration_days = (last_message_date - first_message_date).days + 1
    
    messages_per_day = total_messages / duration_days if duration_days > 0 else 0
    
    media_messages = df['has_media'].sum()
    media_percentage = (media_messages / total_messages) * 100 if total_messages > 0 else 0
    
    urls_sent = df['has_url'].sum()
    
    total_emojis = df['emoji_count'].sum()
    avg_message_length = df['message_length'].mean()
    
    stats = {
        'Total Messages': total_messages,
        'Total Users': total_users,
        'First Message': first_message_date.strftime('%Y-%m-%d'),
        'Last Message': last_message_date.strftime('%Y-%m-%d'),
        'Chat Duration (days)': duration_days,
        'Messages per Day': round(messages_per_day, 2),
        'Media Messages': int(media_messages),
        'Media Messages (%)': round(media_percentage, 2),
        'URLs Shared': int(urls_sent),
        'Total Emojis': int(total_emojis),
        'Average Message Length': round(avg_message_length, 2)
    }
    
    return stats

def get_user_stats(df):
    """Calculate stats for each user."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Messages per user
    user_message_counts = df['user'].value_counts().reset_index()
    user_message_counts.columns = ['User', 'Message Count']
    
    # Calculate percentage of total messages
    total_messages = len(df)
    user_message_counts['Percentage'] = (user_message_counts['Message Count'] / total_messages * 100).round(2)
    
    # Calculate average message length per user
    avg_length = df.groupby('user')['message_length'].mean().reset_index()
    avg_length.columns = ['User', 'Average Length']
    
    # Media messages per user
    media_counts = df[df['has_media']].groupby('user').size().reset_index(name='Media Count')
    
    # Emoji counts per user
    emoji_counts = df.groupby('user')['emoji_count'].sum().reset_index()
    emoji_counts.columns = ['User', 'Emoji Count']
    
    # Merge all stats
    user_stats = user_message_counts.merge(avg_length, on='User', how='left')
    user_stats = user_stats.merge(media_counts, left_on='User', right_on='user', how='left').drop(columns=['user'])
    user_stats = user_stats.merge(emoji_counts, on='User', how='left')
    
    # Fill NaN values with 0
    user_stats = user_stats.fillna(0)
    
    # Round average length
    user_stats['Average Length'] = user_stats['Average Length'].round(2)
    
    # Calculate messages per day
    chat_duration = (df['timestamp'].max() - df['timestamp'].min()).days + 1
    if chat_duration > 0:
        user_stats['Messages/Day'] = (user_stats['Message Count'] / chat_duration).round(2)
    else:
        user_stats['Messages/Day'] = user_stats['Message Count']
    
    return user_stats

def get_time_analysis(df):
    """Analyze message patterns by time (hour, day, month)."""
    if df is None or df.empty:
        return {}, {}, {}
    
    # Messages by hour
    hourly_messages = df.groupby('hour').size().reset_index(name='count')
    hourly_dict = dict(zip(hourly_messages['hour'], hourly_messages['count']))
    
    # Fill missing hours with 0
    for hour in range(24):
        if hour not in hourly_dict:
            hourly_dict[hour] = 0
    
    # Sort by hour
    hourly_dict = {k: hourly_dict[k] for k in sorted(hourly_dict.keys())}
    
    # Messages by day of week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_messages = df.groupby('day').size().reset_index(name='count')
    daily_dict = dict(zip(daily_messages['day'], daily_messages['count']))
    
    # Fill missing days with 0
    for day in days_order:
        if day not in daily_dict:
            daily_dict[day] = 0
    
    # Sort by day of week
    daily_dict = {day: daily_dict[day] for day in days_order if day in daily_dict}
    
    # Messages by month
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
    monthly_messages = df.groupby('month').size().reset_index(name='count')
    monthly_dict = dict(zip(monthly_messages['month'], monthly_messages['count']))
    
    # Fill missing months with 0
    for month in months_order:
        if month not in monthly_dict:
            monthly_dict[month] = 0
    
    # Sort by month
    monthly_dict = {month: monthly_dict[month] for month in months_order if month in monthly_dict}
    
    return hourly_dict, daily_dict, monthly_dict

def get_emoji_analysis(df):
    """Analyze emoji usage."""
    if df is None or df.empty or 'emojis' not in df.columns:
        return {}, {}
    
    # Flatten list of emojis
    all_emojis = []
    for emoji_list in df['emojis']:
        all_emojis.extend(emoji_list)
    
    # Count emojis
    emoji_counts = Counter(all_emojis)
    
    # Get top emojis
    top_emojis = dict(emoji_counts.most_common(20))
    
    # Emoji usage by user
    user_emoji_counts = {}
    for user in df['user'].unique():
        user_emojis = []
        for emoji_list in df[df['user'] == user]['emojis']:
            user_emojis.extend(emoji_list)
        
        if user_emojis:
            user_emoji_counts[user] = dict(Counter(user_emojis).most_common(10))
        else:
            user_emoji_counts[user] = {}
    
    return top_emojis, user_emoji_counts

def get_sentiment_analysis(df):
    """Analyze sentiment of messages."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original dataframe
    sentiment_df = df.copy()
    
    # Define a function to calculate sentiment
    def calculate_sentiment(text):
        if pd.isna(text) or text == "<Media omitted>":
            return 0
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    # Apply sentiment analysis to each message
    sentiment_df['sentiment'] = sentiment_df['message'].apply(calculate_sentiment)
    
    # Categorize sentiment
    def categorize_sentiment(score):
        if score > 0.3:
            return 'Very Positive'
        elif score > 0:
            return 'Positive'
        elif score == 0:
            return 'Neutral'
        elif score > -0.3:
            return 'Negative'
        else:
            return 'Very Negative'
    
    sentiment_df['sentiment_category'] = sentiment_df['sentiment'].apply(categorize_sentiment)
    
    # Calculate average sentiment per user
    user_sentiment = sentiment_df.groupby('user')['sentiment'].mean().reset_index()
    user_sentiment.columns = ['User', 'Average Sentiment']
    user_sentiment['Sentiment Category'] = user_sentiment['Average Sentiment'].apply(categorize_sentiment)
    user_sentiment['Average Sentiment'] = user_sentiment['Average Sentiment'].round(3)
    
    return user_sentiment

# def get_activity_timeline(df):
#     """Create a daily activity timeline."""
#     if df is None or df.empty:
#         return pd.DataFrame()
    
#     # Count messages per day
#     timeline = df.groupby('date').size().reset_index(name='count')
    
#     # Ensure continuous date range
#     date_range = pd.date_range(start=timeline['date'].min(), end=timeline['date'].max())
#     date_df = pd.DataFrame({'date': date_range})
    
#     # Merge with actual counts
#     timeline = date_df.merge(timeline, on='date', how='left').fillna(0)
#     timeline['count'] = timeline['count'].astype(int)
    
#     return timeline


def get_activity_timeline(df):
    """Create a daily activity timeline."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Count messages per day
    timeline = df.groupby('date').size().reset_index(name='count')
    
    # Ensure continuous date range
    date_range = pd.date_range(start=timeline['date'].min(), end=timeline['date'].max())
    date_df = pd.DataFrame({'date': date_range})
    
    # Convert both 'date' columns to datetime64[ns]
    date_df['date'] = pd.to_datetime(date_df['date'])
    timeline['date'] = pd.to_datetime(timeline['date'])
    
    # Merge with actual counts
    timeline = date_df.merge(timeline, on='date', how='left').fillna(0)
    timeline['count'] = timeline['count'].astype(int)
    
    return timeline


def get_response_patterns(df):
    """Analyze response patterns between users."""
    if df is None or df.empty or len(df) < 2:
        return {}
    
    # Sort by timestamp to ensure messages are in order
    sorted_df = df.sort_values('timestamp')
    
    # Create a column with the next message's user
    sorted_df['next_user'] = sorted_df['user'].shift(-1)
    
    # Count response patterns
    response_patterns = sorted_df.groupby(['user', 'next_user']).size().reset_index(name='count')
    
    # Filter out self-responses (same user posting consecutive messages)
    response_patterns = response_patterns[response_patterns['user'] != response_patterns['next_user']]
    
    # Convert to dictionary for easier use
    patterns_dict = {}
    for _, row in response_patterns.iterrows():
        if row['user'] not in patterns_dict:
            patterns_dict[row['user']] = {}
        patterns_dict[row['user']][row['next_user']] = int(row['count'])
    
    return patterns_dict

def get_response_times(df):
    """Calculate average response times between users."""
    if df is None or df.empty or len(df) < 2:
        return {}
    
    # Sort by timestamp to ensure messages are in order
    sorted_df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Create columns for previous message's user and timestamp
    sorted_df['prev_user'] = sorted_df['user'].shift(1)
    sorted_df['prev_timestamp'] = sorted_df['timestamp'].shift(1)
    
    # Calculate time difference in minutes
    sorted_df['response_time'] = (sorted_df['timestamp'] - sorted_df['prev_timestamp']).dt.total_seconds() / 60
    
    # Filter out self-responses and response times > 24 hours (1440 minutes)
    response_df = sorted_df[(sorted_df['user'] != sorted_df['prev_user']) & 
                           (sorted_df['response_time'] <= 1440) &
                           (sorted_df['response_time'] > 0)]
    
    # Calculate average response time for each user pair
    response_times = response_df.groupby(['prev_user', 'user'])['response_time'].mean().reset_index()
    response_times.columns = ['From', 'To', 'Avg Response Time (min)']
    response_times['Avg Response Time (min)'] = response_times['Avg Response Time (min)'].round(2)
    
    return response_times

def get_word_cloud_data(df, user=None):
    """Prepare data for word cloud visualization."""
    if df is None or df.empty:
        return ""
    
    if user:
        filtered_df = df[df['user'] == user]
    else:
        filtered_df = df
    
    # Combine all messages
    all_text = ' '.join(filtered_df['message'].tolist())
    
    # Remove URLs
    all_text = re.sub(r'https?://\S+', '', all_text)
    
    # Remove media messages
    all_text = re.sub(r'<Media omitted>|image omitted|video omitted|sticker omitted', '', all_text, flags=re.IGNORECASE)
    
    return all_text

def get_chat_intensity(df):
    """Calculate chat intensity (messages per day) over time."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Count messages per day
    daily_counts = df.groupby('date').size().reset_index(name='count')
    
    # Calculate 7-day rolling average
    daily_counts['rolling_avg'] = daily_counts['count'].rolling(window=7, min_periods=1).mean().round(2)
    
    return daily_counts

def get_user_participation_over_time(df):
    """Analyze how user participation changes over time."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Create month-year column for grouping
    df['month_year'] = df['timestamp'].dt.strftime('%Y-%m')
    
    # Count messages per user per month
    monthly_participation = df.groupby(['month_year', 'user']).size().reset_index(name='count')
    
    pivot_table = monthly_participation.pivot_table(
        index='month_year', 
        columns='user', 
        values='count',
        fill_value=0
    ).reset_index()
 
    pivot_table = pivot_table.sort_values('month_year')
    
    return pivot_table
