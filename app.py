import os
import json
import numpy as np
import gradio as gr
from PIL import Image
from model_loader import load_model, preprocess_image, predict_disease
from chat_app import groq_chatbot

# Load the disease diagnosis model
model_path = "attached_assets/mobilenetv2.h5"
model = load_model(model_path)

# Load class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Disease treatments dictionary
DEMO_TREATMENTS = {
    "Apple - Apple Scab": "Rake and destroy fallen leaves, prune for good air circulation, apply fungicides like captan or sulfur before rainy periods, and plant resistant apple varieties.",
    "Apple - Black Rot": "Prune and remove infected branches, destroy fallen leaves and fruit, apply copper-based fungicides, and ensure proper spacing between trees.",
    "Tomato - Early Blight": "Remove infected leaves, apply fungicides containing chlorothalonil or copper, and mulch around plants to prevent soil splashing.",
    "Tomato - Healthy": "Your tomato plant is healthy! Maintain regular watering, ensure adequate sunlight, and monitor for pests or diseases.",
    "Background without Leaves": "No plants detected. Please upload an image containing leaves to diagnose."
}

# Diagnosis function
def diagnose_image(image):
    if image is None:
        return "⚠️ Please upload an image for diagnosis."
    
    try:
        img_array = np.array(image)
        preprocessed_img = preprocess_image(img_array)
        disease_label, confidence = predict_disease(model, preprocessed_img, class_labels)
        confidence_pct = f"{confidence:.1f}%"
        
        treatment = DEMO_TREATMENTS.get(
            disease_label, 
            "No specific treatment information available for this condition. Consult with an agricultural expert."
        )
        
        result = f"### 🌿 Diagnosis: {disease_label.replace('_', ' ')}\n"
        result += f"**Confidence:** {confidence_pct}\n\n"
        result += f"### 🛠 Recommended Treatment:\n{treatment}"
        return result
    except Exception as e:
        return f"❌ Error during diagnosis: {e}"

# Build Gradio UI
with gr.Blocks(css="footer {visibility: hidden}") as app:
    gr.Markdown("# 🌱 Plant Disease Diagnosis & Agricultural Chatbot")
    gr.Markdown("Upload an image of a plant leaf for disease detection, or ask the chatbot for agricultural advice.")

    with gr.Row():
        # LEFT: Image Upload & Disease Diagnosis
        with gr.Column(scale=1):
            gr.Markdown("## 📸 Upload Image for Diagnosis")
            image_input = gr.Image(type="numpy", label="Upload Leaf Image")
            diagnose_button = gr.Button("🔍 Diagnose", variant="primary")
            diagnosis_output = gr.Markdown(label="Diagnosis Results")

            diagnose_button.click(fn=diagnose_image, inputs=[image_input], outputs=[diagnosis_output])

        # RIGHT: Chatbot for Agricultural Advice
        with gr.Column(scale=1):
            gr.Markdown("## 🤖 Ask the Agricultural Chatbot")
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="Ask a question about agriculture...", label="Your Question")
            clear = gr.Button("🗑 Clear Chat")
            chat_history_state = gr.State([])

            msg.submit(fn=groq_chatbot, inputs=[msg, chat_history_state], outputs=[chatbot, msg])
            clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

    gr.Markdown("---")
    gr.Markdown("### ℹ️ About this Application")
    gr.Markdown("This AI-powered tool helps diagnose plant diseases and provides treatment recommendations. It also includes a chatbot for general agricultural queries.")

# Launch Gradio app
if __name__ == "__main__":
    app.launch(share=True)
