import os
from groq import Groq
import gradio as gr

# Load API key from environment variable
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable. Set it in Hugging Face Spaces Secrets.")

client = Groq(api_key=API_KEY)

VALIDATION_PROMPT = "You are an intelligent assistant. Analyze the input question carefully. Respond with 'Yes' if the input is agriculture-related, and 'No' otherwise."
RESPONSE_PROMPT = "You are an agriculture expert. Provide a concise and accurate answer to the following agriculture-related question:"

def validate_input(input_text):
    try:
        validation_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": VALIDATION_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
            max_completion_tokens=1,
        )
        return validation_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def get_agriculture_response(input_text):
    try:
        detailed_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": RESPONSE_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0.5,
            max_completion_tokens=250,
        )
        return detailed_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def groq_chatbot(input_text, chat_history):
    validation_result = validate_input(input_text)
    if validation_result.lower() == "yes":
        response = get_agriculture_response(input_text)
        chat_history.append((input_text, response))  # Store the chat history as tuples
        return chat_history, response
    elif validation_result.lower() == "no":
        return chat_history, "❌ This is not an agriculture-related question."
    else:
        return chat_history, f"⚠️ Error: Unexpected response: {validation_result}"

def launch_gradio_interface():
    with gr.Blocks(css="style.css") as demo:
        gr.Markdown("Ask questions about plant diseases, treatments, or general agricultural topics.")
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(placeholder="Ask a question about agriculture...", label="Your Question")
        clear = gr.Button("Clear Chat")
        chat_history_state = gr.State([])  # Maintain chat history
        
        msg.submit(fn=groq_chatbot, inputs=[msg, chat_history_state], outputs=[chat_history_state, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)

    demo.launch(share=True)

if __name__ == "__main__":
    launch_gradio_interface()
