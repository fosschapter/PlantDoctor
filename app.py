import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from model_loader import load_model, preprocess_image, predict_disease
import gradio as gr

# Import RAG functionality
from rag import RAGModel

# Load the RAG model
rag_model = RAGModel(model_path="path_to_rag_model", retriever_path="path_to_retriever")

# Load the CNN model and class labels
model_path = "attached_assets/mobilenetv2.h5"
model = load_model(model_path)

with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Demo treatment information
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

# Function for image diagnosis
def diagnose_image(image):
    if image is None:
        return "Please upload an image for diagnosis."
    
    # Convert gradio image to numpy array
    if isinstance(image, np.ndarray):
        img_array = image
    else:
        # For PIL Image or other formats
        img = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        img_array = np.array(img)
    
    # Process image and make prediction
    preprocessed_img = preprocess_image(img_array)
    disease_label, confidence = predict_disease(model, preprocessed_img, class_labels)
    
    # Format the confidence percentage
    confidence_pct = f"{confidence:.1f}%"
    
    # Get treatment recommendation if available
    treatment = DEMO_TREATMENTS.get(disease_label, 
                                 "No specific treatment information available for this condition. Consider consulting with a local agricultural extension service.")
    
    result = f"### Diagnosis: {disease_label.replace('_', ' ')}\n\n"
    result += f"### Confidence: {confidence_pct}\n\n"
    result += f"### Recommended Treatment:\n{treatment}"
    
    return result

# Function for agriculture chatbot (RAG integrated)
def chat_with_bot(message, history):
    if not message:
        return "Please ask a question about plant diseases or treatments."
    
    # Use RAG model to generate response
    response = rag_model.generate_response(message, history)
    
    # Return response
    return response

# Create Gradio Interface
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
        
        with gr.TabItem("Agricultural Chatbot"):
            gr.Markdown("Ask questions about plant diseases, treatments, or general agricultural topics.")
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="Ask a question about agriculture...", label="Your Question")
            clear = gr.Button("Clear Chat")
            
            msg.submit(fn=chat_with_bot, inputs=[msg, chatbot], outputs=[chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
    
    gr.Markdown("## About this Application")
    gr.Markdown("""
    This application uses a MobileNetV2 model trained on the PlantVillage dataset to diagnose common plant diseases from leaf images.
    
    The chatbot provides information about various agricultural topics, plant diseases, and treatments using a RAG-based model for accurate and contextual responses.
    
    **Note:** This is a simplified version designed to work in environments like Hugging Face Spaces.
    
    Image upload + Diagnosis → Get recommendations → Chat for more information
    """)

# For Hugging Face Spaces compatibility
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False)
