import os
from groq import Groq
import gradio as gr

# Initialize the Groq client with your API key
client = Groq(api_key="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")  # Replace with your actual API key

# Define the validation prompt
VALIDATION_PROMPT = """
You are an intelligent assistant. Analyze the input question carefully.
Respond with "Yes" if the input is agriculture-related, and "No" otherwise. Provide only "Yes" or "No" as your answer, nothing else.
"""

# Define a generic response prompt
RESPONSE_PROMPT = """
You are an agriculture expert. Provide a concise and accurate answer to the following agriculture-related question:
"""

# Function to validate and process the input
def validate_and_process(input_text):
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

        # Extract the LLM's response for validation
        response_text = validation_response.choices[0].message.content.strip()

        # Debugging: Print the response for verification
        print(f"Validation Response: {response_text}")

        # Step 2: Check the response and process accordingly
        if response_text.lower() == "yes":
            # If valid, pass the question to the LLM for an answer
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

            # Extract the answer from the LLM
            answer = detailed_response.choices[0].message.content.strip()
            return f"Valid Agriculture Input:\n{answer}"

        elif response_text.lower() == "no":
            # If not agriculture-related, return a message
            return f"Invalid Input: Not agriculture-related. Please ask questions specifically about agriculture."

        else:
            return f"Unexpected Response from LLM: {response_text}"

    except Exception as e:
        # Handle any errors gracefully
        return f"Error: {e}"

"""
# Create a Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Agriculture-Based Input Validator and Question Answerer")

    input_text = gr.Textbox(label="Enter your question:")
    output_text = gr.Textbox(label="Validation Result and Answer:")

    validate_button = gr.Button("Validate and Process")

    validate_button.click(
        fn=validate_and_process,
        inputs=input_text,
        outputs=output_text,
    )

demo.launch()
"""