import os
import json
import numpy as np
import requests
import gradio as gr
from PIL import Image
from model_loader import load_model, preprocess_image, predict_disease
from chat_app import groq_chatbot

# OpenWeatherMap API Key
OWM_API_KEY = "aeacce09a78f91e055b86b8ffb349d4f"

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
    "Tomato - Healthy": "Your tomato plant is healthy! Maintain regular watering, ensure adequate sunlight, and monitor for pests or diseases."
}

# Function to get latitude & longitude
def get_coordinates(location):
    """Fetch latitude & longitude for a given location"""
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OWM_API_KEY}"
    response = requests.get(url).json()
    
    if not response:
        return None, None  # Invalid location
    
    return response[0]["lat"], response[0]["lon"]

# Function to get weather & air quality
def get_weather_and_aqi(location):
    """Fetch weather & air quality index (AQI)"""
    lat, lon = get_coordinates(location)
    if lat is None:
        return "❌ Invalid Location"

    # Weather API
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
    weather_res = requests.get(weather_url).json()

    # AQI API
    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OWM_API_KEY}"
    aqi_res = requests.get(aqi_url).json()

    if "main" not in weather_res or "list" not in aqi_res:
        return "❌ Data Unavailable"

    aqi_level = aqi_res["list"][0]["main"]["aqi"]  # AQI scale: 1 (Good) → 5 (Hazardous)
    aqi_labels = ["🟢 Good", "🟡 Fair", "🟠 Moderate", "🔴 Poor", "🟣 Hazardous"]
    
    return (
        f"🌍 **{location}**\n"
        f"🌡️ Temp: **{weather_res['main']['temp']}°C**\n"
        f"💧 Humidity: **{weather_res['main']['humidity']}%**\n"
        f"🌫️ AQI: **{aqi_labels[aqi_level - 1]} ({aqi_level})**"
    )

# Function to diagnose plant disease
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
            "No specific treatment information available. Consult with an agricultural expert."
        )
        
        result = f"### 🌿 Diagnosis: {disease_label.replace('_', ' ')}\n"
        result += f"**Confidence:** {confidence_pct}\n\n"
        result += f"### 🛠 Recommended Treatment:\n{treatment}"
        return result
    except Exception as e:
        return f"❌ Error during diagnosis: {e}"

# Build Gradio UI
with gr.Blocks(css="footer {visibility: hidden}") as app:
    gr.Markdown("# 🌱 AI-Powered Agricultural Assistant")
    gr.Markdown("Upload an image for plant disease detection, ask a chatbot for agricultural advice, or check local weather & air quality.")

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

    # Weather & AQI Section
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🌤️ Weather & Air Quality")
            location_dropdown = gr.Textbox(placeholder="Enter City Name", label="🌍 Location")
            weather_output = gr.Markdown(label="🌦️ Weather & AQI")
            fetch_button = gr.Button("🔄 Update Weather")

            fetch_button.click(get_weather_and_aqi, inputs=location_dropdown, outputs=weather_output)

    gr.Markdown("---")
    gr.Markdown("### ℹ️ About this Application")
    gr.Markdown("This AI-powered tool helps diagnose plant diseases, provides weather insights, and includes a chatbot for agricultural queries.")

# Launch Gradio app
if __name__ == "__main__":
    app.launch(share=True)
