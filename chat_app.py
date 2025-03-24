import gradio as gr
from groq import Groq

# Initialize the Groq client with your API key
client = Groq(api_key="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")  # Replace with your API key

# Function to check if the input is related to agriculture using the LLM
def is_agriculture_related(user_input):
    # Prompt the LLM to determine if the input is agriculture-related
    try:
        check_message = [
            {"role": "system", "content": "You are an AI assistant. Respond with 'yes' if the following input is related to agriculture, otherwise respond with 'no'."},
            {"role": "user", "content": user_input}
        ]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Replace with your specific model
            messages=check_message,
            temperature=0,  # Use deterministic output
            max_completion_tokens=1,
            top_p=1,
            stop=None,
        )

        # Parse the LLM response
        decision = response.choices[0].message["content"].strip().lower()
        print(decision)
        return decision == "yes" or "'yes'"
    except Exception as e:
        print(f"Error checking input relevance: {e}")
        return False

# Function to handle user input and get a response from the Groq API
def groq_chatbot(user_input, chat_history=[]):
    try:
        # Check if the input is agriculture-related using the LLM
        if not is_agriculture_related(user_input):
            # If not related to agriculture, respond accordingly
            bot_reply = "Please ask something related to agriculture."
            chat_history.append((user_input, bot_reply))
            return "", chat_history

        # Prepare messages for the Groq API
        messages = [{"role": "system", "content": "You are a helpful assistant specialized in agriculture."}]
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
    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error: {str(e)}"
        chat_history.append((user_input, error_message))
        return "", chat_history
