import os
from groq import Groq
import gradio as gr

# Load API key from environment variable
API_KEY = os.getenv("GROQ_API_KEY")  # Set this in Hugging Face Spaces Secrets
if not API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable. Set it in Hugging Face Spaces Secrets.")

# Initialize Groq client
client = Groq(api_key=API_KEY)

# Define prompts
VALIDATION_PROMPT = "You are an intelligent assistant. Analyze the input question carefully. Respond with 'Yes' if the input is agriculture-related, and 'No' otherwise."
RESPONSE_PROMPT = "You are an agriculture expert. Provide a concise and accurate answer to the following agriculture-related question:"

def validate_input(input_text):
    """Checks if input is agriculture-related."""
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
    """Generates an answer if the input is agriculture-related."""
    try:
        detailed_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": RESPONSE_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0.5,
            max_completion_tokens=150,
        )
        return detailed_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def groq_chatbot(input_text):
    """Validates input and processes it."""
    validation_result = validate_input(input_text)
    if validation_result.lower() == "yes":
        response = get_agriculture_response(input_text)
        return "Valid", response
    elif validation_result.lower() == "no":
        return "Invalid", "Invalid Input: Not agriculture-related. Please ask agriculture-specific questions."
    else:
        return "Error", f"Unexpected Response from LLM: {validation_result}"

def launch_gradio_interface():
    """Launch Gradio UI."""
    with gr.Blocks() as demo:
        gr.Markdown("### Agriculture-Based Input Validator and Question Answerer")
        input_text = gr.Textbox(label="Enter your question:")
        output_text = gr.Textbox(label="Validation Result and Answer:")
        validate_button = gr.Button("Validate and Process")
        validate_button.click(fn=lambda x: groq_chatbot(x)[1], inputs=input_text, outputs=output_text)

    demo.launch(share=True)  # Remove `server_name="0.0.0.0"`

if __name__ == "__main__":
    launch_gradio_interface()
