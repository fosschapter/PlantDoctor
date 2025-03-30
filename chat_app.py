import os
from groq import Groq
import gradio as gr

# Load API key
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable. Set it in Hugging Face Spaces Secrets.")

client = Groq(api_key=API_KEY)

VALIDATION_PROMPT = "You are an intelligent assistant. Analyze the input image carefully. Respond with 'Yes' if it is agriculture-related, and 'No' otherwise."
RESPONSE_PROMPT = "You are an agriculture expert. Provide a concise and accurate answer about the given agriculture-related image."

def validate_image(image):
    """Checks if the uploaded image is agriculture-related."""
    try:
        validation_response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": VALIDATION_PROMPT},
                {"role": "user", "content": {"type": "image", "image_url": image}},
            ],
            temperature=0,
            max_completion_tokens=1,
        )
        return validation_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def analyze_agriculture_image(image):
    """Generates an answer if the image is agriculture-related."""
    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": RESPONSE_PROMPT},
                {"role": "user", "content": {"type": "image", "image_url": image}},
            ],
            temperature=0.5,
            max_completion_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def process_image(image):
    """Validates the image and processes it."""
    validation_result = validate_image(image)
    if validation_result.lower() == "yes":
        response = analyze_agriculture_image(image)
        return "✅ Valid Agriculture Image", response
    elif validation_result.lower() == "no":
        return "❌ Invalid Image", "This image does not appear to be agriculture-related. Please upload a relevant image."
    else:
        return "⚠️ Error", f"Unexpected Response: {validation_result}"

def launch_gradio_interface():
    """Launch Gradio UI for image-based validation."""
    with gr.Blocks(css="style.css") as demo:
        gr.Markdown(
            """
            <h1 style="text-align: center; color: #4CAF50;">🌾 Agriculture AI Image Analyzer</h1>
            <p style="text-align: center; font-size: 18px;">Upload an image, and the AI will check if it is agriculture-related and provide an analysis.</p>
            """,
        )

        image_input = gr.Image(label="Upload an agriculture-related image", type="filepath")
        validate_button = gr.Button("Validate & Process", elem_id="validate-button")
        
        validation_result = gr.Textbox(label="Validation", interactive=False, elem_id="validation-result")
        output_text = gr.Textbox(label="AI Response", interactive=False, elem_id="output-box")

        # Allow both button click and pressing "Enter" to submit
        image_input.upload(fn=process_image, inputs=image_input, outputs=[validation_result, output_text])
        validate_button.click(fn=process_image, inputs=image_input, outputs=[validation_result, output_text])

    demo.launch(share=True)

if __name__ == "__main__":
    launch_gradio_interface()
