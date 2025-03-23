import os
import json
import numpy as np
from PIL import Image
from model_loader import load_model, preprocess_image, predict_disease
import gradio as gr
from groq import Groq
import tensorflow as tf

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
    "Apple - Cedar Apple Rust": "Remove nearby juniper or cedar trees if possible, apply fungicides like myclobutanil, and plant resistant apple varieties.",
    "Apple - Healthy": "Your apple tree appears healthy! Continue regular maintenance, including pruning, watering, and monitoring for pests or disease symptoms.",
    "Background without Leaves": "No plants detected. Please upload an image containing leaves to diagnose.",
    "Blueberry - Healthy": "Your blueberry plant is healthy! Ensure consistent watering, proper mulching, and protect it from frost during early spring.",
    "Cherry - Powdery Mildew": "Remove infected leaves, avoid overhead watering, ensure good air circulation, and apply fungicides containing sulfur or potassium bicarbonate.",
    "Cherry - Healthy": "Your cherry tree is healthy! Continue providing proper care, including regular pruning, watering, and monitoring for pests or diseases.",
    "Corn - Cercospora Leaf Spot (Gray Leaf Spot)": "Rotate crops annually, apply fungicides like strobilurins or triazoles, and ensure good field drainage and proper plant spacing.",
    "Corn - Common Rust": "Apply fungicides containing propiconazole or azoxystrobin, plant resistant corn varieties, and avoid overhead irrigation.",
    "Corn - Northern Leaf Blight": "Rotate crops, plant resistant varieties, and apply fungicides like mancozeb or strobilurin when symptoms first appear.",
    "Corn - Healthy": "Your corn plant looks healthy! Continue to monitor for any signs of disease and ensure proper spacing for airflow.",
    "Grape - Black Rot": "Remove mummified berries and infected leaves, prune for good air circulation, and apply fungicides such as myclobutanil or captan.",
    "Grape - Esca (Black Measles)": "Prune and destroy infected parts, practice proper vineyard sanitation, and avoid mechanical injuries to the vines.",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)": "Remove and destroy infected leaves, apply fungicides containing copper, and ensure proper spacing for air circulation.",
    "Grape - Healthy": "Your grapevine is healthy! Maintain regular pruning, proper watering, and monitoring for pests or diseases.",
    "Orange - Huanglongbing (Citrus Greening)": "Unfortunately, there is no cure. Remove and destroy infected trees, control psyllid populations using insecticides, and plant disease-free certified saplings.",
    "Peach - Bacterial Spot": "Apply copper-based bactericides, remove and destroy infected leaves and fruit, and plant resistant peach varieties.",
    "Peach - Healthy": "Your peach tree looks healthy! Continue regular care, including pruning, fertilizing, and monitoring for pests or diseases.",
    "Pepper (Bell) - Bacterial Spot": "Apply fixed copper sprays, avoid overhead irrigation, remove infected plants, and practice crop rotation.",
    "Pepper (Bell) - Healthy": "Your bell pepper plant looks healthy! Ensure adequate sunlight, watering, and keep monitoring for any pests or diseases.",
    "Potato - Early Blight": "Remove infected leaves, apply fungicides with active ingredients like chlorothalonil, and ensure proper spacing between plants.",
    "Potato - Late Blight": "Apply fungicides like mancozeb or chlorothalonil, remove and destroy infected plants, and avoid overhead watering.",
    "Potato - Healthy": "Your potato plant looks healthy! Maintain proper watering, ensure good soil drainage, and monitor for any signs of disease.",
    "Raspberry - Healthy": "Your raspberry plant is healthy! Ensure proper support, regular pruning, and protect it from pests and harsh weather conditions.",
    "Soybean - Healthy": "Your soybean crop is healthy! Monitor regularly for any signs of disease or pests, and maintain proper crop rotation.",
    "Squash - Powdery Mildew": "Apply fungicides with sulfur or potassium bicarbonate, remove infected leaves, and ensure good air circulation.",
    "Strawberry - Leaf Scorch": "Remove infected leaves, avoid overhead watering, and apply fungicides like captan or mancozeb.",
    "Strawberry - Healthy": "Your strawberry plants are healthy! Maintain consistent watering, ensure good air circulation, and protect from frost.",
    "Tomato - Bacterial Spot": "Remove infected leaves, apply fixed copper sprays, and avoid overhead irrigation. Ensure proper plant spacing.",
    "Tomato - Early Blight": "Remove infected leaves, apply fungicides containing chlorothalonil or copper, and mulch around plants to prevent soil splashing.",
    "Tomato - Late Blight": "Apply fungicides like chlorothalonil or mancozeb, remove infected plants, and ensure proper spacing for airflow.",
    "Tomato - Leaf Mold": "Remove infected leaves, ensure proper air circulation, and apply fungicides containing chlorothalonil or copper.",
    "Tomato - Septoria Leaf Spot": "Remove and destroy infected leaves, apply fungicides like mancozeb, and avoid overhead watering.",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)": "Spray the undersides of leaves with water, apply horticultural oils or insecticidal soaps, and maintain humidity around plants.",
    "Tomato - Target Spot": "Remove infected leaves, apply fungicides like chlorothalonil, and ensure proper spacing between plants for air circulation.",
    "Tomato - Tomato Yellow Leaf Curl Virus": "Remove infected plants, control whitefly populations with insecticides, and plant resistant tomato varieties.",
    "Tomato - Tomato Mosaic Virus": "Remove infected plants, sterilize tools, and avoid handling plants when wet.",
    "Tomato - Healthy": "Your tomato plant is healthy! Maintain regular watering, ensure adequate sunlight, and monitor for pests or diseases."
}


# Diagnosis function
def diagnose_image(image):
    if image is None:
        return "Please upload an image for diagnosis."
    
    try:
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
    except Exception as e:
        return f"Error during diagnosis: {e}"

# Groq Chatbot API configuration
client = Groq(api_key="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")

# Chatbot function using Groq API
def chat_with_bot(message, history):
    if not message.strip():
        return history + [["", "Please ask a question about plant diseases or treatments."]]

    try:
        # Add a system prompt to guide the chatbot
        messages = [
            {"role": "system", "content": "You are an expert in plant diseases and treatments. Answer questions based on agricultural knowledge."},
            {"role": "user", "content": message}
        ]

        # Query the Groq API
        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=messages,
            temperature=0.5,
            max_completion_tokens=250,
            top_p=1.0,
            stream=True,
        )

        # Collect response from the stream
        response = ""
        for chunk in completion:
            if "choices" in chunk and chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content

        if not response.strip():
            response = "I couldn't generate an answer. Please try rephrasing your question or ask a different one."

        history.append([message, response])
    except Exception as e:
        history.append([message, f"Error communicating with Groq API: {e}"])

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
        
        with gr.TabItem("Agricultural Chatbot"):
            gr.Markdown("Ask questions about plant diseases, treatments, or general agricultural topics.")
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="Ask a question about agriculture...", label="Your Question")
            clear = gr.Button("Clear Chat")
    
            msg.submit(fn=chat_with_bot, inputs=[msg, chatbot], outputs=[chatbot])
            clear.click(lambda: [], None, chatbot, queue=False)
    
    gr.Markdown("## About this Application")
    gr.Markdown("""
    This application uses a MobileNetV2 model trained on the PlantVillage dataset to diagnose common plant diseases from leaf images.
    The chatbot provides dynamic answers using the Groq API for question-answering.
    """)
    
# Launch Gradio app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=True)