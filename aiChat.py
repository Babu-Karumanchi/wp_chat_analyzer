import google.generativeai as genai



GENAI_API_KEY = "AIzaSyC8Oi6OM5GFr7iUO6pDIMldR__WUp0nJRg" 
genai.configure(api_key=GENAI_API_KEY)



def get_chat_summary(chat_text):
    model = genai.GenerativeModel('models/gemini-1.5-flash')  
    prompt = f"""
    Analyze the following WhatsApp chat and provide last 3-2 months a summary including:
    - Main discussion topics
    - Important announcements
    - Trends (e.g., exams, assignments, events)

    Chat data:
    {chat_text} 
    """
    response = model.generate_content(prompt)
    return response.text if response else "No summary available."

def ask_gemini_question(chat_text, query):
    model = genai.GenerativeModel("models/gemini-1.5-flash") 
    prompt = f"""
    Based on the following WhatsApp chat, answer this query concisely:
    
    Chat Data:
    {chat_text[:]}

    Query: {query}
    """
    response = model.generate_content(prompt)
    return response.text if response else "No response available."