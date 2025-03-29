import gradio as gr
from groq import Groq
from transformers import pipeline

# Initialize the Groq client with your API key
client = Groq(api_key="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")

# For a more accurate agriculture classification, use a zero-shot classifier
# This allows you to specify custom categories without needing a pre-trained agriculture classifier
classifier = pipeline("zero-shot-classification", 
                     model="facebook/bart-large-mnli")

def classify_agriculture_intent(query):
    """Classify if the query is related to agriculture using zero-shot classification."""
    categories = ["agriculture", "plants", "farming", "crops", "plant disease", "gardening", 
                 "plant health", "cultivation", "irrigation", "soil", "fertilizer", "harvest"]
    
    result = classifier(query, categories, multi_label=True)
    
    # If any agriculture category has a confidence above threshold, consider it agricultural
    # Adjust the threshold based on testing
    threshold = 0.35
    for label, score in zip(result['labels'], result['scores']):
        if score > threshold:
            return True
    
    return False

def groq_chatbot(user_input, chat_history=[]):
    try:
        # Check if the query is agriculture-related
        is_agriculture = classify_agriculture_intent(user_input)
        
        if is_agriculture:
            # Prepare messages for the Groq API with agriculture-specific system prompt
            messages = [{"role": "system", "content": "You are a plant and agriculture expert. Provide helpful, accurate information about plant diseases, farming techniques, and agricultural practices. Focus on sustainable solutions."}]
            
            for user_message, bot_message in chat_history:
                messages.append({"role": "user", "content": user_message})
                messages.append({"role": "assistant", "content": bot_message})
            
            messages.append({"role": "user", "content": user_input})
            
            # Call the Groq API
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,  # Lowered for more factual responses
                max_completion_tokens=300,
                top_p=1,
                stream=True,
                stop=None,
            )
            
            # Collect the response from the streaming output
            bot_reply = ""
            for chunk in completion:
                bot_reply += chunk.choices[0].delta.content or ""
        else:
            # Polite response for non-agriculture questions
            bot_reply = "I'm specialized in helping with agricultural and plant-related questions. Could you please ask me something about plants, farming, or plant diseases? I'd be happy to assist with those topics."
        
        # Update the chat history
        chat_history.append((user_input, bot_reply))
        return "", chat_history
    
    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error: {str(e)}"
        chat_history.append((user_input, error_message))
        return "", chat_history

# Create the Gradio interface (assuming this is part of your full app.py)
def create_chat_interface():
    with gr.Blocks() as chat_interface:
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(placeholder="Ask me about plants, farming, or plant diseases...", label="Your Question")
        
        msg.submit(groq_chatbot, [msg, chatbot], [msg, chatbot])
        
    return chat_interface