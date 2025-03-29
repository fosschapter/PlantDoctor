import gradio as gr
from groq import Groq
from transformers import pipeline

# Initialize the Groq client
client = Groq(api_key="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")

# Initialize the intent classifier
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
AGRICULTURE_KEYWORDS = ["plant", "crop", "agriculture", "farm", "disease", "pest", "soil", "harvest"]

def is_agriculture_related(query):
    """Check if the query is related to agriculture using keyword matching and classifier"""
    # First check with simple keyword matching (faster)
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in AGRICULTURE_KEYWORDS):
        return True
    
    # If no keywords found, use the classifier
    result = classifier(query)
    # The model returns LABEL_0 or LABEL_1 - we'll consider LABEL_1 as positive
    return result[0]['label'] == 'LABEL_1' and result[0]['score'] > 0.7

def groq_chatbot(user_input, chat_history=[]):
    try:
        # Check if the question is agriculture-related
        if not is_agriculture_related(user_input):
            bot_reply = "I specialize in plant health and agriculture. Please ask me about plant diseases, crop care, or farming practices."
            chat_history.append((user_input, bot_reply))
            return "", chat_history
        
        # Prepare messages for the Groq API (original logic for agriculture questions)
        messages = [{"role": "system", "content": "You are an agricultural expert assistant specialized in plant diseases and farming practices. Provide concise, practical advice."}]
        
        for user_message, bot_message in chat_history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_message})
        
        messages.append({"role": "user", "content": user_input})
        
        # Call the Groq API
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=1,
            max_completion_tokens=250,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        # Collect the response from the streaming output
        bot_reply = ""
        for chunk in completion:
            bot_reply += chunk.choices[0].delta.content or ""
        
        # Update the chat history
        chat_history.append((user_input, bot_reply))
        return "", chat_history
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        chat_history.append((user_input, error_message))
        return "", chat_history