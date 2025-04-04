import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import aiChat as chatBot

# Import custom modules
from utils import preprocess_chat, identify_group_members, extract_common_words, identify_chat_type
from analyzer import (
    get_basic_stats, get_user_stats, get_time_analysis, get_emoji_analysis,
    get_sentiment_analysis, get_activity_timeline, get_response_patterns,
    get_response_times, get_word_cloud_data, get_chat_intensity,
    get_user_participation_over_time
)

# Set page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("WhatsApp Chat Analyzer 💬")

# Sidebar
st.sidebar.title("Upload Chat Export")
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp exported chat file (.txt)", type=["txt"])

st.sidebar.markdown("""
### How to export your WhatsApp chat
1. Open the chat you want to analyze
2. Tap the three dots (⋮) in the top right
3. Select 'More' > 'Export chat'
4. Choose 'Without media'
5. Send/save the .txt file
6. Upload the file here
""")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Process uploaded file
if uploaded_file is not None:
    with st.spinner("Processing chat data..."):
        df = preprocess_chat(uploaded_file)
        chat_text = uploaded_file.read().decode("utf-8")
        if df is not None and not df.empty:
            st.session_state.df = df
            st.sidebar.success(f"✅ Chat data loaded successfully!")
            
            # Display basic info in sidebar
            chat_type = identify_chat_type(df)
            members = identify_group_members(df)
            
            st.sidebar.subheader("Chat Info")
            st.sidebar.markdown(f"**Type:** {chat_type}")
            st.sidebar.markdown(f"**Total Messages:** {len(df):,}")
            st.sidebar.markdown(f"**Date Range:** {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
            st.sidebar.markdown(f"**Participants:** {len(members)}")
            
            if len(members) <= 10:  # Only show list if not too many members
                st.sidebar.markdown("**Members:**")
                for member in members:
                    st.sidebar.markdown(f"- {member}")
        else:
            st.error("Failed to process the uploaded file. Please make sure it's a valid WhatsApp chat export.")

# Main content - only show if data is loaded
if st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df
    
    # Create tabs for different analyses
    tabs = st.tabs([
        "📊 Overview", 
        "👤 User Analysis", 
        "⏱️ Time Analysis", 
        "😀 Emoji Analysis",
        "🔤 Word Analysis",
        "🔄 Interaction Patterns",
        "📈 Activity Timeline",
        "❤️ Sentiment Analysis",
        " 🤖 Ai Analysis"
    ])
    
    

    # Overview tab
    with tabs[0]:
        st.header("Chat Overview")
        
        # Display basic stats
        stats = get_basic_stats(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Messages</div>
            </div>
            """.format(stats['Total Messages']), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Chat Duration (days)</div>
            </div>
            """.format(stats['Chat Duration (days)']), unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}</div>
                <div class="metric-label">Messages per Day</div>
            </div>
            """.format(stats['Messages per Day']), unsafe_allow_html=True)
        
        st.subheader("Basic Statistics")
        
        # Create two columns for stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**First Message:** {stats['First Message']}")
            st.markdown(f"**Last Message:** {stats['Last Message']}")
            st.markdown(f"**Total Users:** {stats['Total Users']}")
            st.markdown(f"**Media Messages:** {stats['Media Messages']} ({stats['Media Messages (%)']}%)")
        
        with col2:
            st.markdown(f"**URLs Shared:** {stats['URLs Shared']}")
            st.markdown(f"**Total Emojis:** {stats['Total Emojis']}")
            st.markdown(f"**Average Message Length:** {stats['Average Message Length']} characters")
        
        # Message distribution by user
        st.subheader("Message Distribution by User")
        
        user_stats = get_user_stats(df)
        
        st.write(user_stats.head())
        
        # Create bar chart for message distribution
        fig = px.bar(
            user_stats, 
            x='User', 
            y='Message Count',
            title='Number of Messages by User',
            color='Message Count',
            color_continuous_scale='Viridis',
            text='Percentage'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=500, width=800)
        st.plotly_chart(fig, use_container_width=True)
        
        # Chat intensity over time
        st.subheader("Chat Intensity Over Time")
        
        # Get chat intensity data
        intensity_data = get_chat_intensity(df)
        
        # Create line chart for chat intensity
        fig = px.line(
            intensity_data, 
            x='date', 
            y=['count', 'rolling_avg'],
            title='Messages per Day with 7-day Rolling Average',
            labels={'value': 'Messages', 'date': 'Date', 'variable': 'Metric'},
            color_discrete_map={'count': '#1f77b4', 'rolling_avg': '#ff7f0e'}
        )
        fig.update_layout(height=400, legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Word cloud for the entire chat
        st.subheader("Word Cloud")
        
        word_cloud_text = get_word_cloud_data(df)
        
        if word_cloud_text:
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(word_cloud_text)
            
            # Create matplotlib figure
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            # Display the figure
            st.pyplot(plt)
    
    # User Analysis tab
    with tabs[1]:
        st.header("User Analysis")
        
        # User selection
        all_users = ['All Users'] + list(df['user'].unique())
        selected_user = st.selectbox("Select User", all_users)
        
        # Filter data by selected user
        if selected_user != 'All Users':
            filtered_df = df[df['user'] == selected_user]
            user_title = f"Analysis for {selected_user}"
        else:
            filtered_df = df
            user_title = "Analysis for All Users"
        
        st.subheader(user_title)
        
        # Basic user metrics
        user_metrics = get_user_stats(filtered_df) if selected_user == 'All Users' else get_user_stats(df[df['user'] == selected_user])
        
        if not user_metrics.empty:
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            if selected_user != 'All Users':
                total_messages = user_metrics['Message Count'].values[0]
                avg_length = user_metrics['Average Length'].values[0]
                media_count = user_metrics['Media Count'].values[0] if 'Media Count' in user_metrics else 0
                emoji_count = user_metrics['Emoji Count'].values[0]
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:,}</div>
                        <div class="metric-label">Total Messages</div>
                    </div>
                    """.format(int(total_messages)), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.1f}</div>
                        <div class="metric-label">Avg Message Length</div>
                    </div>
                    """.format(avg_length), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:,}</div>
                        <div class="metric-label">Media Messages</div>
                    </div>
                    """.format(int(media_count)), unsafe_allow_html=True)
                
                with col4:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:,}</div>
                        <div class="metric-label">Emojis Used</div>
                    </div>
                    """.format(int(emoji_count)), unsafe_allow_html=True)
            else:
                # Display table for all users
                st.dataframe(user_metrics, use_container_width=True)
        
        # Word cloud for selected user
        st.subheader(f"Word Cloud for {selected_user if selected_user != 'All Users' else 'All Users'}")
        
        user_word_cloud_text = get_word_cloud_data(filtered_df)
        
        if user_word_cloud_text:
            user_wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(user_word_cloud_text)
            
            # Create matplotlib figure
            plt.figure(figsize=(10, 5))
            plt.imshow(user_wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            # Display the figure
            st.pyplot(plt)
        
        # Message activity over time
        st.subheader("Message Activity Over Time")
        
        # Filter timeline data for selected user
        timeline_data = get_activity_timeline(filtered_df)
        
        if not timeline_data.empty:
            # Create line chart for user activity
            fig = px.line(
                timeline_data, 
                x='date', 
                y='count',
                title=f'Messages per Day for {selected_user if selected_user != "All Users" else "All Users"}',
                labels={'count': 'Messages', 'date': 'Date'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Time Analysis tab
    with tabs[2]:
        st.header("Time Analysis")
        
        # Get time data
        hourly_data, daily_data, monthly_data = get_time_analysis(df)
        
        # Messages by hour
        st.subheader("Messages by Hour of Day")
        
        hour_df = pd.DataFrame({
            'Hour': list(hourly_data.keys()),
            'Messages': list(hourly_data.values())
        })
        
        # Create bar chart for hourly distribution
        fig = px.bar(
            hour_df, 
            x='Hour', 
            y='Messages',
            title='Message Distribution by Hour',
            color='Messages',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Messages by day of week
        st.subheader("Messages by Day of Week")
        
        day_df = pd.DataFrame({
            'Day': list(daily_data.keys()),
            'Messages': list(daily_data.values())
        })
        
        # Create bar chart for daily distribution
        fig = px.bar(
            day_df, 
            x='Day', 
            y='Messages',
            title='Message Distribution by Day of Week',
            color='Messages',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Messages by month
        st.subheader("Messages by Month")
        
        month_df = pd.DataFrame({
            'Month': list(monthly_data.keys()),
            'Messages': list(monthly_data.values())
        })
        
        # Create bar chart for monthly distribution
        fig = px.bar(
            month_df, 
            x='Month', 
            y='Messages',
            title='Message Distribution by Month',
            color='Messages',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Emoji Analysis tab
    with tabs[3]:
        st.header("Emoji Analysis")
        
        # Get emoji data
        top_emojis, user_emojis = get_emoji_analysis(df)
        
        if top_emojis:
            # Display top emojis
            st.subheader("Top Emojis Used in Chat")
            
            emoji_df = pd.DataFrame({
                'Emoji': list(top_emojis.keys()),
                'Count': list(top_emojis.values())
            })
            
            # Create bar chart for emoji distribution
            fig = px.bar(
                emoji_df, 
                x='Emoji', 
                y='Count',
                title='Top Emojis in Chat',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display top emojis by user
            st.subheader("Top Emojis by User")
            
            # User selection for emoji analysis
            emoji_user = st.selectbox("Select User for Emoji Analysis", ['All Users'] + list(df['user'].unique()))
            
            if emoji_user != 'All Users':
                if emoji_user in user_emojis and user_emojis[emoji_user]:
                    user_emoji_df = pd.DataFrame({
                        'Emoji': list(user_emojis[emoji_user].keys()),
                        'Count': list(user_emojis[emoji_user].values())
                    })
                    
                    # Create bar chart for user emoji distribution
                    fig = px.bar(
                        user_emoji_df, 
                        x='Emoji', 
                        y='Count',
                        title=f'Top Emojis for {emoji_user}',
                        color='Count',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"{emoji_user} did not use any emojis.")
            else:
                # Create tabs for each user's emoji data
                emoji_tabs = st.tabs(list(user_emojis.keys()))
                
                for i, user in enumerate(user_emojis.keys()):
                    with emoji_tabs[i]:
                        if user_emojis[user]:
                            user_emoji_df = pd.DataFrame({
                                'Emoji': list(user_emojis[user].keys()),
                                'Count': list(user_emojis[user].values())
                            })
                            
                            # Create bar chart for user emoji distribution
                            fig = px.bar(
                                user_emoji_df, 
                                x='Emoji', 
                                y='Count',
                                title=f'Top Emojis for {user}',
                                color='Count',
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"{user} did not use any emojis.")
        else:
            st.info("No emojis were found in the chat.")
    
    # Word Analysis tab
    with tabs[4]:
        st.header("Word Analysis")
        
        # Get common words
        common_words = extract_common_words(df)
        
        if common_words:
            # Display top words
            st.subheader("Most Common Words in Chat")
            
            word_df = pd.DataFrame({
                'Word': list(common_words.keys()),
                'Count': list(common_words.values())
            })
            
            # Create bar chart for word distribution
            fig = px.bar(
                word_df.head(20), 
                x='Word', 
                y='Count',
                title='Top 20 Words in Chat',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Word analysis by user
            st.subheader("Word Analysis by User")
            
            # User selection for word analysis
            word_user = st.selectbox("Select User for Word Analysis", ['All Users'] + list(df['user'].unique()))
            
            if word_user != 'All Users':
                user_words = extract_common_words(df[df['user'] == word_user])
                
                if user_words:
                    user_word_df = pd.DataFrame({
                        'Word': list(user_words.keys()),
                        'Count': list(user_words.values())
                    })
                    
                    # Create bar chart for user word distribution
                    fig = px.bar(
                        user_word_df.head(20), 
                        x='Word', 
                        y='Count',
                        title=f'Top 20 Words for {word_user}',
                        color='Count',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No common words found for {word_user}.")
        else:
            st.info("Could not extract common words from the chat.")
    
    # Interaction Patterns tab
    with tabs[5]:
        st.header("Interaction Patterns")
        
        # Get response patterns
        response_patterns = get_response_patterns(df)
        response_times = get_response_times(df)
        
        if response_patterns:
            # Display response patterns
            st.subheader("Who Responds to Whom")
            
            # Convert to dataframe for visualization
            patterns_list = []
            for user1, responses in response_patterns.items():
                for user2, count in responses.items():
                    patterns_list.append({
                        'From': user1,
                        'To': user2,
                        'Count': count
                    })
            
            patterns_df = pd.DataFrame(patterns_list)
            
            # Create heatmap for response patterns
            # Pivot the dataframe
            pivot_df = patterns_df.pivot(index='From', columns='To', values='Count').fillna(0)
            
            fig = px.imshow(
                pivot_df,
                title='Response Patterns Between Users',
                labels=dict(x='Responder', y='Initial Sender', color='Message Count'),
                color_continuous_scale='Viridis',
                aspect='auto'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        if not response_times.empty:
            # Display response times
            st.subheader("Average Response Times Between Users")
            
            st.dataframe(response_times, use_container_width=True)
            
            # Create a bar chart for response times
            fig = px.bar(
                response_times,
                x='From',
                y='Avg Response Time (min)',
                color='To',
                title='Average Response Time by User',
                barmode='group'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # User participation over time
        st.subheader("User Participation Over Time")
        
        participation_data = get_user_participation_over_time(df)
        
        if not participation_data.empty:
            # Melt the dataframe for visualization
            melted_df = pd.melt(
                participation_data,
                id_vars=['month_year'],
                var_name='User',
                value_name='Messages'
            )
            
            # Create line chart for user participation over time
            fig = px.line(
                melted_df,
                x='month_year',
                y='Messages',
                color='User',
                title='User Participation Over Time',
                labels={'month_year': 'Month', 'Messages': 'Message Count'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Activity Timeline tab
    with tabs[6]:
        st.header("Activity Timeline")
        
        # Get activity timeline
        timeline_data = get_activity_timeline(df)
        
        if not timeline_data.empty:
            # Create line chart for activity timeline
            fig = px.line(
                timeline_data,
                x='date',
                y='count',
                title='Messages per Day',
                labels={'count': 'Messages', 'date': 'Date'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add calendar heatmap
            st.subheader("Message Activity Calendar")
            
            # Add year selector
            years = sorted(timeline_data['date'].dt.year.unique())
            selected_year = st.selectbox("Select Year", years, index=len(years)-1)
            
            # Filter data for selected year
            year_data = timeline_data[timeline_data['date'].dt.year == selected_year]
            
            # Create dataframe for heatmap
            calendar_df = year_data.copy()
            calendar_df['day'] = calendar_df['date'].dt.day
            calendar_df['month'] = calendar_df['date'].dt.month
            
            # Create pivot table
            pivot_data = calendar_df.pivot_table(
                index='day',
                columns='month',
                values='count',
                aggfunc='sum'
            ).fillna(0)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                pivot_data,
                cmap='Greens',
                linewidths=.5,
                ax=ax,
                cbar_kws={'label': 'Messages'}
            )
            
            # Set labels
            ax.set_title(f'Message Activity Calendar ({selected_year})')
            ax.set_xlabel('Month')
            ax.set_ylabel('Day')
            
            # Set x-axis labels to month names
            # ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            
            ax.set_xticks(range(12))  # Set ticks for each month (0 to 11)
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Month names

            # Rotate the labels to make them readable
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
    
    # Sentiment Analysis tab
    with tabs[7]:
        st.header("Sentiment Analysis")
        
        # Get sentiment data
        sentiment_data = get_sentiment_analysis(df)
        
        if not sentiment_data.empty:
            # Display sentiment stats
            st.subheader("Sentiment Analysis by User")
            
            # Format the dataframe
            sentiment_display = sentiment_data.copy()
            sentiment_display['Average Sentiment'] = sentiment_display['Average Sentiment'].apply(lambda x: f"{x:+.3f}")
            
            # Show the dataframe
            st.dataframe(sentiment_display, use_container_width=True)
            
            # Create bar chart for sentiment
            fig = px.bar(
                sentiment_data,
                x='User',
                y='Average Sentiment',
                color='Sentiment Category',
                title='Average Sentiment by User',
                labels={'Average Sentiment': 'Sentiment Score (-1 to +1)'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            
            # Add sentiment column to original dataframe
            sentiment_df = df.copy()
            
            # Calculate sentiment for each message
            from textblob import TextBlob
            
            def calculate_sentiment(text):
                if pd.isna(text) or text == "<Media omitted>":
                    return 0
                analysis = TextBlob(text)
                return analysis.sentiment.polarity
            
            sentiment_df['sentiment'] = sentiment_df['message'].apply(calculate_sentiment)
            
            # Create histogram for sentiment distribution
            fig = px.histogram(
                sentiment_df,
                x='sentiment',
                nbins=20,
                title='Sentiment Distribution',
                labels={'sentiment': 'Sentiment Score (-1 to +1)', 'count': 'Number of Messages'},
                color_discrete_sequence=['#4CAF50']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment over time
            st.subheader("Sentiment Over Time")
            
            # Group by date and calculate average sentiment
            sentiment_timeline = sentiment_df.groupby('date')['sentiment'].mean().reset_index()
            sentiment_timeline['rolling_avg'] = sentiment_timeline['sentiment'].rolling(window=7, min_periods=1).mean()
            
            # Create line chart for sentiment over time
            fig = px.line(
                sentiment_timeline,
                x='date',
                y=['sentiment', 'rolling_avg'],
                title='Sentiment Over Time',
                labels={'value': 'Average Sentiment', 'date': 'Date', 'variable': 'Type'},
                color_discrete_map={'sentiment': '#1f77b4', 'rolling_avg': '#ff7f0e'}
            )
            fig.update_layout(
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                legend_title_text=''
            )
            st.plotly_chart(fig, use_container_width=True)
            

    with tabs[8]:
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []

        if uploaded_file is not None:
            # st.subheader("Chat Summary")

            # AI Chat Summary
            st.subheader("📢 AI-Generated Summary")
            summary = chatBot.get_chat_summary(chat_text)
            # with st.expander("Chat Q&A"):
            #     st.write(summary)
            st.write(summary)

            # Display Question/Answer History
            st.subheader("💬 Question/Answer History")
            if st.session_state.qa_history:
                for qa in st.session_state.qa_history:
                    st.markdown(f"**Question:** {qa['question']}")
                    st.write(f"**Answer:** {qa['answer']}")
                    st.write("---")  # Separator
            else:
                st.write("No questions asked yet.")
            # AI Chat Q&A
            # user_query = st.chat_input("Enter your question about the chat:")
        # selected_tab = None
        # for index, tab in enumerate(tabs):
        #     with tab:
        #         if index == 8:  # 8th tab = "🤖 Ai Analysis"
        #             selected_tab = "Ai Analysis"

        # # Show chat input only in the "🤖 Ai Analysis" tab
        # if selected_tab == "Ai Analysis":
        #     user_query = st.chat_input("Enter your question about the AI Analysis:")
        #     if user_query:
        #         st.write("You asked:", user_query)
            # if user_query:
            #     st.write("You asked:", user_query)
            
            user_query = st.text_input("Enter your question:")


            if user_query: # Only do this with the user_query.
                answer = chatBot.ask_gemini_question(chat_text, user_query)
                # st.write(answer)

                # Store the question and answer in the session state
                st.session_state.qa_history.append({"question": user_query, "answer": answer})

   
            
else:
    # Show landing page if no data is loaded
    st.markdown("""
    ## 📱 WhatsApp Chat Analyzer
    
    Upload your WhatsApp chat export file to get started!
    
    ### Features:
    - **Basic Statistics**: Messages, media, URLs, emojis, etc.
    - **User Analysis**: Compare activity between users
    - **Time Analysis**: When are people most active?
    - **Emoji Analysis**: Most used emojis by person and overall
    - **Word Analysis**: Discover the most common words
    - **Interaction Patterns**: See who responds to whom
    - **Activity Timeline**: Track chat activity over time
    - **Sentiment Analysis**: Measure the mood of conversations
    - **Ai Analysis**: Ask the AI questions about the chat
    
    ### How to get started:
    1. Export your WhatsApp chat (without media)
    2. Upload the .txt file using the sidebar
    3. Explore your chat data through the different analysis tabs
    """)
    
    # Add sample image
    # st.image("wp.jpg", 
    #          caption="Sample WhatsApp Chat Analysis", use_column_width=True)
