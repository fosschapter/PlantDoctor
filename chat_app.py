import os
import gradio as gr
from groq import GroqClient

# Initialize the Groq client with the API key from environment variables
api_key = os.getenv("GROQ_API_KEY")  # Replace with your environment variable
client = GroqClient(api_key=api_key)

# Function to handle user input and get a response from the Groq API
def groq_chatbot(user_input, chat_history=[]):
    try:
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
            temperature=0.7,
            max_completion_tokens=250,
            top_p=0.95,
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
        print(f"Error: {e}")
        error_message = "An error occurred. Please try again."
        chat_history.append((user_input, error_message))
        return "", chat_history

# Create the Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your message here...")
    clear = gr.Button("Clear")

    # Connect the components
    msg.submit(groq_chatbot, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot)

demo.launch()
