import os
from groq import Groq
import gradio as gr

# Initialize the Groq client with your API key
API_KEY = os.getenv("GROQ_API_KEY")  # Set this in Hugging Face Spaces Secrets
if not API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable. Set it in Hugging Face Spaces Secrets.")

client = Groq(api_key=API_KEY)

# Define the validation prompt
VALIDATION_PROMPT = """
You are an intelligent assistant. Analyze the input question carefully.
Respond with "Yes" if the input is agriculture-related, and "No" otherwise. Provide only "Yes" or "No" as your answer, nothing else.
"""

# Define a generic response prompt
RESPONSE_PROMPT = """
You are an agriculture expert. Provide a concise and accurate answer to the following agriculture-related question:
"""

def validate_input(input_text):
    """
    Validates whether the given input is agriculture-related or not.

    Parameters:
        input_text (str): The input question.

    Returns:
        str: "Yes" if agriculture-related, "No" otherwise.
    """
    try:
        # Step 1: Send the validation prompt to the LLM
        validation_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Replace with your specific model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": VALIDATION_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
            max_completion_tokens=1,
        )
        # Extract and return the validation result
        return validation_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def get_agriculture_response(input_text):
    """
    Gets a detailed response for a valid agriculture-related question.

    Parameters:
        input_text (str): The agriculture-related question.

    Returns:
        str: The response to the question.
    """
    try:
        # Step 2: Send the agriculture question to the LLM
        detailed_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Replace with your specific model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": RESPONSE_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0.5,
            max_completion_tokens=250,
        )
        # Extract and return the response
        return detailed_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def groq_chatbot(input_text):
    """
    Validates the input and processes it to return a relevant response.

    Parameters:
        input_text (str): The input question.

    Returns:
        tuple: A tuple containing the validation status and the final message.
    """
    validation_result = validate_input(input_text)
    if validation_result.lower() == "yes":
        response = get_agriculture_response(input_text)
        return "Valid", response
    elif validation_result.lower() == "no":
        return "Invalid", "Invalid Input: Not agriculture-related. Please ask questions specifically about agriculture."
    else:
        return "Error", f"Unexpected Response from LLM: {validation_result}"


# Gradio interface for testing
def launch_gradio_interface():

   # Launches the Gradio interface for validating and answering agriculture-related questions.

    with gr.Blocks() as demo:
        gr.Markdown("### Agriculture-Based Input Validator and Question Answerer")

        input_text = gr.Textbox(label="Enter your question:")
        output_text = gr.Textbox(label="Validation Result and Answer:")

        validate_button = gr.Button("Validate and Process")

        validate_button.click(
            fn=lambda x: groq_chatbot(x)[1],  # Call only the result message
            inputs=input_text,
            outputs=output_text,
        )

    demo.launch(server_name="0.0.0.0", share=True)
