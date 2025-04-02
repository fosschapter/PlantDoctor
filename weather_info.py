import requests
import gradio as gr

# OpenWeatherMap API Key
OWM_API_KEY = "aeacce09a78f91e055b86b8ffb349d4f"

def get_coordinates(location):
    """Fetch latitude & longitude for a location"""
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OWM_API_KEY}"
    response = requests.get(url).json()
    
    if not response:
        return None, None  # Invalid location
    
    return response[0]["lat"], response[0]["lon"]

def get_weather_and_aqi(location):
    """Fetch both Weather & AQI in one function"""
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

# Gradio UI
location_dropdown = gr.Dropdown(
    ["New York", "Los Angeles", "London", "Delhi", "Tokyo"], label="🌍 Select Location"
)
weather_output = gr.Markdown(label="🌦️ Weather & AQI")
fetch_button = gr.Button("🔄 Update Weather")

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            location_dropdown.render()
        with gr.Column(scale=2):
            weather_output.render()
        with gr.Column(scale=1):
            fetch_button.render()

    # Update weather when button is clicked
    fetch_button.click(get_weather_and_aqi, inputs=location_dropdown, outputs=weather_output)

demo.launch()
