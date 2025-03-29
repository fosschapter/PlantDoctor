import gradio as gr
from groq import Groq
from transformers import pipeline

# Initialize the Groq client with your API key
client = Groq(api_key="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")  # Replace with your API key

# Load a pre-trained classification pipeline for intent classification
classifier = pipeline("text-classification", model="distilbert-base-uncased")
relevant_category = "Agriculture"

def classify_intent(query):
    """Classify the intent of the query as Agriculture or Other."""
    result = classifier(query)
    label = result[0]['label']
    return relevant_category if label == relevant_category else "Other"

# Fallback replies in case the Groq API is unavailable
fallback_replies = {
    "default": "The Groq API is currently unavailable. Please try again later.",
    "agriculture_related": "This question seems to be related to agriculture. Unfortunately, I can't process it right now.",
    "not_related": "I'm here to assist with agriculture-related queries only. Please ask a relevant question."
}

# Function to handle user input and get a response from the Groq API
def groq_chatbot(user_input, chat_history=[]):
    try:
        # Classify the intent of the query
        intent = classify_intent(user_input)

        if intent == "Agriculture":
            # Prepare messages for the Groq API
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for user_message, bot_message in chat_history:
                messages.append({"role": "user", "content": user_message})
                messages.append({"role": "assistant", "content": bot_message})
            messages.append({"role": "user", "content": user_input})

            # Call the Groq API
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Replace with your specific model
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

        else:
            # If the query is not agriculture-related
            bot_reply = fallback_replies["not_related"]
            chat_history.append((user_input, bot_reply))
            return "", chat_history

    except Exception as e:
        # Handle errors gracefully
        error_message = fallback_replies["default"]
        chat_history.append((user_input, error_message))
        return "", chat_history
