# Use a slim Python base
FROM python:3.10-slim

# Metadata
LABEL maintainer="Rayean Patric <patricrayean@gmail.com>"
WORKDIR /app

# Install git (to clone HF Space) and venv support
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        && rm -rf /var/lib/apt/lists/*

# Clone your Hugging Face Space
RUN git clone https://github.com/rayeanpatric/PlantDoctor.git ./

# Create & activate a virtualenv, install Python deps
RUN python -m venv env \
    && . env/bin/activate \
    && pip install --no-cache-dir -r requirements.txt

# Expose Gradio default port
EXPOSE 7860

# Allow API keys to be passed at runtime
ENV GROQ_API_KEY=""
ENV OPENWEATHER_API_KEY=""

# Activate venv and launch your app
CMD ["/bin/bash", "-lc", "source env/bin/activate && python app.py"]
