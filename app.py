import os
import json
import numpy as np
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import gradio as gr
from model_loader import load_model, preprocess_image, predict_disease

# Load the disease diagnosis model
model_path = "attached_assets/mobilenetv2.h5"
model = load_model(model_path)

# Load class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Disease treatments dictionary
DEMO_TREATMENTS = {
    "Tomato_Late_blight": "For late blight in tomatoes, remove and destroy infected plants, apply copper-based fungicides, ensure good air circulation, and consider resistant varieties for future plantings.",
    "Tomato_Early_blight": "For early blight in tomatoes, remove infected leaves, apply fungicides containing chlorothalonil or copper, mulch around plants, and practice crop rotation.",
    "Apple_scab": "For apple scab, rake and destroy fallen leaves, prune for air circulation, apply fungicides (like captan or sulfur) before rainy periods, and consider resistant apple varieties.",
    "Tomato_healthy": "Your tomato plant appears healthy! Maintain regular watering, ensure adequate sunlight, and continue monitoring for any signs of disease.",
    "Apple_healthy": "Your apple tree appears healthy! Continue with regular maintenance, including proper pruning, watering, and monitoring for pests or disease symptoms.",
    "Potato_Early_blight": "For early blight in potatoes, remove infected leaves, apply suitable fungicides, ensure adequate plant spacing, and practice crop rotation.",
    "Corn_Common_rust": "For common rust in corn, apply fungicides with active ingredients like propiconazole or azoxystrobin, plant resistant varieties, and avoid overhead irrigation.",
    "Grape_Black_rot": "For black rot in grapes, remove mummified berries, prune for good air circulation, apply fungicides like myclobutanil or captan, and maintain a clean vineyard floor."
}

# Diagnosis function
def diagnose_image(image):
    if image is None:
        return "Please upload an image for diagnosis."
    
    img_array = np.array(image)
    preprocessed_img = preprocess_image(img_array)
    disease_label, confidence = predict_disease(model, preprocessed_img, class_labels)
    confidence_pct = f"{confidence:.1f}%"
    
    treatment = DEMO_TREATMENTS.get(
        disease_label, 
        "No specific treatment information available for this condition. Consult with an agricultural expert."
    )
    
    result = f"### Diagnosis: {disease_label.replace('_', ' ')}\n\n"
    result += f"### Confidence: {confidence_pct}\n\n"
    result += f"### Recommended Treatment:\n{treatment}"
    return result

# Load a lightweight LLM for the chatbot
model_name = "distilbert-base-uncased-distilled-squad"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

def chat_with_bot(message, history):
    if not message:
        return history + [["", "Please ask a question about plant diseases or treatments."]]

    # Dynamic context construction
    relevant_context = """
    This chatbot is knowledgeable about plant diseases, treatments, and general agricultural practices. It uses AI to provide helpful insights based on your questions.
    """
    
    # Search for matching disease/treatment advice in DEMO_TREATMENTS
    for disease, treatment in DEMO_TREATMENTS.items():
        if disease.lower().replace("_", " ") in message.lower():
            relevant_context += f"\n\nTreatment for {disease.replace('_', ' ')}: {treatment}"
            break  # Add advice for the first match

    try:
        # Use the pipeline for answering questions
        response = qa_pipeline(question=message, context=relevant_context)
        answer = response["answer"]
    except Exception as e:
        answer = f"Sorry, I couldn't process your request. Error: {str(e)}"

    # Append the user message and the bot's response to the history
    history.append([message, answer])
    return history


# Gradio interface
with gr.Blocks(title="Plant Disease Diagnosis and Treatment", css="footer {visibility: hidden}") as app:
    gr.Markdown("# 🌱 Plant Disease Diagnosis and Treatment")
    gr.Markdown("Upload a leaf image to diagnose plant diseases and get treatment recommendations.")
    
    with gr.Tabs():
        with gr.TabItem("Diagnose Plant Disease"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy", label="Upload Leaf Image")
                    diagnose_button = gr.Button("Diagnose Disease", variant="primary")
                with gr.Column():
                    diagnosis_output = gr.Markdown(label="Diagnosis Results")
            
            diagnose_button.click(fn=diagnose_image, inputs=[image_input], outputs=[diagnosis_output])
        
        # Update the Gradio TabItem for the chatbot
        with gr.TabItem("Agricultural Chatbot"):
            gr.Markdown("Ask questions about plant diseases, treatments, or general agricultural topics.")
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="Ask a question about agriculture...", label="Your Question")
            clear = gr.Button("Clear Chat")
    
            # Pass the history explicitly to the `chat_with_bot` function
            msg.submit(fn=chat_with_bot, inputs=[msg, chatbot], outputs=[chatbot])
            clear.click(lambda: [], None, chatbot, queue=False)
    
    gr.Markdown("## About this Application")
    gr.Markdown("""
    This application uses a MobileNetV2 model trained on the PlantVillage dataset to diagnose common plant diseases from leaf images.
    The chatbot provides dynamic answers using a lightweight LLM for question-answering.
    """)
    
# Launch Gradio app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False)
