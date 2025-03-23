import os
import json
import tempfile
import numpy as np
from PIL import Image
import tensorflow as tf
from model_loader import load_model, preprocess_image, predict_disease
import gradio as gr

# Load the model and class labels
model_path = "attached_assets/mobilenetv2.h5"
model = load_model(model_path)

# Load class labels
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

# Function for agriculture chatbot
def chat_with_bot(message, history):
    if not message:
        return "Please ask a question about plant diseases or treatments."
    
    # Enhanced keyword-based responses for the demo
    message = message.lower()
    
    # Specific plant disease questions
    if "tomato" in message and "blight" in message:
        if "late" in message:
            response = DEMO_TREATMENTS["Tomato_Late_blight"]
        elif "early" in message:
            response = DEMO_TREATMENTS["Tomato_Early_blight"]
        else:
            response = "There are different types of tomato blight. Common ones include Early Blight and Late Blight. Each requires different treatment approaches."
    elif "apple" in message and "scab" in message:
        response = DEMO_TREATMENTS["Apple_scab"]
    
    # General agriculture questions
    elif "what is agriculture" in message or "define agriculture" in message:
        response = "Agriculture is the science and practice of farming, including the cultivation of soil for growing crops and raising animals to provide food, fiber, and other products. It forms the foundation of our food systems and has been central to human civilization for thousands of years."
    
    elif "sustainable agriculture" in message or "sustainable farming" in message:
        response = "Sustainable agriculture involves farming practices that protect the environment, public health, and animal welfare while ensuring economic viability. This includes techniques such as crop rotation, reduced tillage, precision agriculture, and integrated pest management."
    
    elif "organic farming" in message:
        response = "Organic farming is an agricultural method that relies on natural processes and materials instead of synthetic chemicals. It emphasizes soil health, biodiversity, and ecological balance. Organic farmers avoid synthetic pesticides, fertilizers, genetically modified organisms, and growth hormones."
    
    elif "crop rotation" in message:
        response = "Crop rotation is the practice of growing different types of crops in the same area across sequential seasons. It helps to reduce soil erosion, increase soil fertility and crop yield, and control pests, weeds, and diseases."
    
    elif "hydroponics" in message:
        response = "Hydroponics is a method of growing plants without soil, using mineral nutrient solutions in a water solvent. Plants can be grown with their roots in the mineral nutrient solution only, or in an inert medium such as perlite or gravel. This technique is often used in controlled environments like greenhouses."

    elif "vertical farming" in message:
        response = "Vertical farming is the practice of growing crops in vertically stacked layers, often incorporating controlled-environment agriculture which aims to optimize plant growth. It often uses soilless farming techniques such as hydroponics, aquaponics, and aeroponics, and can be practiced in buildings, shipping containers, or repurposed warehouses."
    
    # Plant care topics
    elif "healthy" in message:
        response = "Healthy plants should be maintained with proper watering, sunlight, and nutrition. Regular inspection for early signs of disease is also important. Maintaining appropriate spacing between plants ensures good air circulation, which helps prevent fungal diseases."
    
    elif any(word in message for word in ["fertilizer", "fertilize", "nutrient"]):
        response = "Most plants benefit from balanced fertilizers with nitrogen, phosphorus, and potassium. The exact ratio depends on the plant type and growth stage. Organic options include compost, manure, and bone meal. Applying fertilizer at the right time and in the right amount is crucial - too much can damage plants and pollute water sources."
    
    elif any(word in message for word in ["water", "watering", "irrigation"]):
        response = "Proper watering is essential for plant health. Most plants prefer deep, infrequent watering rather than frequent shallow watering. Always check soil moisture before watering. Watering early in the morning reduces evaporation and allows foliage to dry before evening, which helps prevent disease."
    
    elif any(word in message for word in ["organic", "natural", "pesticide", "insecticide"]):
        response = "Organic pest control methods include neem oil, insecticidal soap, diatomaceous earth, and beneficial insects like ladybugs. Cultural practices like crop rotation and companion planting can also help prevent pest problems. Creating habitat for beneficial insects and birds is another natural way to control pest populations."
    
    elif "soil health" in message or "soil quality" in message:
        response = "Soil health is vital for plant growth and productivity. Healthy soil has good structure, adequate organic matter, beneficial microorganisms, and proper nutrient balance. Practices that improve soil health include adding compost, avoiding overworking the soil, planting cover crops, and minimizing chemical inputs."
    
    elif "composting" in message:
        response = "Composting is the natural process of recycling organic material like leaves, food scraps, and yard waste into a rich soil amendment. Good compost needs a balance of 'green' materials (high in nitrogen) and 'brown' materials (high in carbon), plus adequate moisture and aeration for the microorganisms that break down the materials."
    
    # Climate and season-related questions
    elif "climate change" in message and "agriculture" in message:
        response = "Climate change poses significant challenges to agriculture, including more frequent extreme weather events, shifting growing seasons, and changing pest and disease patterns. Adaptation strategies include developing drought-resistant crops, improving irrigation efficiency, diversifying crops, and implementing climate-smart practices."
    
    elif "growing season" in message:
        response = "The growing season is the part of the year during which local conditions (temperature, rainfall, daylight) permit normal plant growth. It varies by location, climate, and the specific requirements of different plants. Understanding your local growing season is essential for planning when to plant and harvest crops."
    
    # Catch-all for other questions
    else:
        response = "That's an interesting agricultural question. Agriculture encompasses many aspects including plant science, animal husbandry, soil management, pest control, and sustainable practices. For more detailed information on this specific topic, you might consider consulting agricultural extension services or specialized resources."
    
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
    
    The chatbot provides information about various agricultural topics, plant diseases, and treatments based on keyword matching.
    
    **Note:** This is a simplified version designed to work in environments like Hugging Face Spaces.
    
    Image upload + Diagnosis → Get recommendations → Chat for more information
    """)

# For Hugging Face Spaces compatibility
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False)