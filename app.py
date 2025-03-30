import os
import json
import numpy as np
from PIL import Image
from model_loader import load_model, preprocess_image, predict_disease
import gradio as gr
from groq import Groq
from chat_app import groq_chatbot
from bing_image_downloader import downloader
import shutil

# Load the MobileNetV2 model
model_path = "attached_assets/mobilenetv2.h5"
model = load_model(model_path)

# Load class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Define plant categories
PLANT_CATEGORIES = [
    "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach",
    "Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"
]

# Ensure directory exists
SAVE_DIR = "healthy_leaves"
os.makedirs(SAVE_DIR, exist_ok=True)

# Download and move healthy leaf images (Run once)
for plant in PLANT_CATEGORIES:
    image_folder = os.path.join(SAVE_DIR, f"{plant}_healthy_leaf")
    if not os.path.exists(image_folder):
        downloader.download(f"{plant} healthy leaf", limit=1, output_dir=SAVE_DIR, adult_filter_off=True, force_replace=False, timeout=60)

    # Move the image to the main folder
    downloaded_folder = os.path.join(SAVE_DIR, f"{plant}_healthy_leaf")
    if os.path.exists(downloaded_folder):
        images = os.listdir(downloaded_folder)
        if images:
            shutil.move(os.path.join(downloaded_folder, images[0]), os.path.join(SAVE_DIR, f"{plant}.jpg"))
        shutil.rmtree(downloaded_folder)  # Remove empty subfolder

# Disease treatments dictionary
DEMO_TREATMENTS = {
    "Apple - Apple Scab": "Rake and destroy fallen leaves, prune for good air circulation, apply fungicides like captan or sulfur.",
    "Apple - Healthy": "Your apple tree appears healthy! Continue regular maintenance, including pruning and watering.",
    "Corn - Healthy": "Your corn plant looks healthy! Continue to monitor for any signs of disease.",
    "Tomato - Late Blight": "Apply fungicides like chlorothalonil or mancozeb, remove infected plants, and ensure proper spacing for airflow.",
    "Tomato - Healthy": "Your tomato plant is healthy! Maintain regular watering and monitor for pests.",
}

# Diagnosis function
def diagnose_image(image):
    if image is None:
        return "Please upload an image for diagnosis.", None

    try:
        img_array = np.array(image)
        preprocessed_img = preprocess_image(img_array)
        disease_label, confidence = predict_disease(model, preprocessed_img, class_labels)
        confidence_pct = f"{confidence:.1f}%"

        # Extract plant name
        plant_name = disease_label.split(" - ")[0]
        treatment = DEMO_TREATMENTS.get(disease_label, "No specific treatment available. Consult an expert.")

        # Load the healthy leaf image
        healthy_leaf_path = os.path.join(SAVE_DIR, f"{plant_name}.jpg")
        healthy_leaf = Image.open(healthy_leaf_path) if os.path.exists(healthy_leaf_path) else None

        result = f"### Diagnosis: {disease_label}\n\n"
        result += f"### Confidence: {confidence_pct}\n\n"
        result += f"### Recommended Treatment:\n{treatment}"

        return result, gr.Image.update(value=healthy_leaf if healthy_leaf else None)
    except Exception as e:
        return f"Error during diagnosis: {e}", None

# Chatbot handler
def chatbot_handler(user_input, chat_history):
    response = groq_chatbot(user_input, chat_history)
    chat_history.append((user_input, response))
    return "", chat_history  # Return empty input and updated history

# Gradio UI
with gr.Blocks(title="Plant Disease Diagnosis and Treatment") as app:
    gr.Markdown("# 🌱 Plant Disease Diagnosis and Treatment")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="📤 Upload Leaf Image")
            diagnose_button = gr.Button("Diagnose Disease", variant="primary")

        with gr.Column():
            healthy_leaf_output = gr.Image(label="🌿 Healthy Leaf", interactive=False)

        with gr.Column():
            chatbot = gr.Chatbot(label="Agriculture Assistant", height=400)
            msg = gr.Textbox(placeholder="Ask about agriculture...", label="Your Question")
            clear = gr.Button("Clear Chat")
            chat_history_state = gr.State([])

    diagnosis_output = gr.Markdown(label="📋 Diagnosis Results")

    # Button Click Actions
    diagnose_button.click(fn=diagnose_image, inputs=[image_input], outputs=[diagnosis_output, healthy_leaf_output])
    msg.submit(fn=chatbot_handler, inputs=[msg, chat_history_state], outputs=[msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

app.launch(server_name="0.0.0.0", share=True)
