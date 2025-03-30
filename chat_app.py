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

def groq_chatbot(input_text):
    validation_result = validate_input(input_text)
    if validation_result.lower() == "yes":
        response = get_agriculture_response(input_text)
        return "✅ Valid Input", response
    elif validation_result.lower() == "no":
        return "❌ Invalid Input", "This is not an agriculture-related question. Please ask something relevant to agriculture."
    else:
        return "⚠️ Error", f"Unexpected response: {validation_result}"

def launch_gradio_interface():
    with gr.Blocks(css="style.css") as demo:
        gr.Markdown(
            """
            <h1 style="text-align: center; color: #4CAF50;">🌱 Agriculture AI Assistant</h1>
            <p style="text-align: center; font-size: 18px;">An AI-powered tool for validating and answering agriculture-related questions.</p>
            """,
            elem_id="title",
        )

        input_text = gr.Textbox(label="Enter your agriculture-related question:", placeholder="Type your question here...", lines=3, elem_id="input-box")
        validate_button = gr.Button("Validate & Process", elem_id="validate-button")
        
        validation_result = gr.Textbox(label="Validation", interactive=False, elem_id="validation-result")
        output_text = gr.Textbox(label="AI Response", interactive=False, elem_id="output-box")

        # Allow both button click and pressing "Enter" to submit
        input_text.submit(fn=groq_chatbot, inputs=input_text, outputs=[validation_result, output_text])
        validate_button.click(fn=groq_chatbot, inputs=input_text, outputs=[validation_result, output_text])

    demo.launch(share=True)

if __name__ == "__main__":
    launch_gradio_interface()
