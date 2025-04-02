import requests

def get_weather(location):
    API_KEY = "YOUR_OPENWEATHER_API_KEY"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()

    weather_data = {
        "temperature": response["main"]["temp"],
        "humidity": response["main"]["humidity"],
        "weather": response["weather"][0]["description"],
    }
    return weather_data

def get_aqi(location):
    API_KEY = "YOUR_IQAIR_API_KEY"
    url = f"http://api.airvisual.com/v2/city?city={location}&key={API_KEY}"
    response = requests.get(url).json()

    return response["data"]["current"]["pollution"]["aqius"]

import gradio as gr

# Function to get live weather and AQI
def get_weather_info(location):
    weather = get_weather(location)
    aqi = get_aqi(location)

    return f"🌍 Location: {location}\n🌡️ Temp: {weather['temperature']}°C\n💧 Humidity: {weather['humidity']}%\n🌫️ AQI: {aqi}"

# Dropdown for selecting the location
location_dropdown = gr.Dropdown(["New York", "Los Angeles", "London", "Delhi", "Tokyo"], label="🌍 Select Location")

# Small UI box for weather details
weather_output = gr.Textbox(label="🌦️ Weather & AQI", interactive=False)

# Button to fetch data
fetch_button = gr.Button("🔄 Update Weather")

# Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            location_dropdown.render()
        with gr.Column(scale=2):
            weather_output.render()
        with gr.Column(scale=1):
            fetch_button.render()

    # Update weather when button is clicked
    fetch_button.click(get_weather_info, inputs=location_dropdown, outputs=weather_output)

demo.launch()

