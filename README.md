# 🌍 PlantDoctor

**A Plant Disease Diagnosis Tool + Agricultural AI Chatbot**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15283721.svg)](https://doi.org/10.5281/zenodo.15283721)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Hugging Face Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-orange)](https://huggingface.co/spaces/rayeanpatric/PlantDoctor)

---

## 🧠 About the Project

**PlantDoctor** is an AI-powered web application designed for rapid **plant disease diagnosis** and intelligent **agricultural query resolution**.

- **Upload leaf images** to detect diseases using a fine-tuned MobileNetV2 model.
- **Receive treatment recommendations** for 15+ plant types.
- **Interact with a chatbot** (powered by `llama-3.3-70b-versatile` via the Groq API) to ask agricultural questions in natural language.

Live demo available on: [Hugging Face Spaces](https://huggingface.co/spaces/rayeanpatric/PlantDoctor)

---

## 🚀 Features

- 🌿 **Leaf-based Disease Detection**
- 💊 **Treatment Recommendation System**
- 🤖 **AI Chatbot for Agricultural Advice**
- 🧪 Built using **Gradio**, **TensorFlow/Keras**, and **Groq LLM API**

---

## 🖥️ Local Installation

> Requires Python 3.10+

```bash
git clone https://github.com/rayeanpatric/PlantDoctor.git
cd PlantDoctor

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

## 🐳 Docker Support

If you'd rather use Docker:

```bash
docker run -it -p 7860:7860 --platform=linux/amd64 -e GROQ_API_KEY="your_value_here" -e OPENWEATHER_API_KEY="your_value_here" registry.hf.space/rayeanpatric-plantdoctor:latest python app.py
```

---

## 📊 Supported Crops

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## 💬 Citing This Work

If you use PlantDoctor in your research, please cite:

```bibtex
@software{rayean_patric_2025_plantdoctor,
  author       = {Rayean Patric F.},
  title        = {PlantDoctor},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15283531},
  url          = {https://doi.org/10.5281/zenodo.15283531}
}
```

---

## 🤝 Contributing

Want to contribute?

1. Fork the repo or [Fork the HF Space](https://huggingface.co/spaces/rayeanpatric/PlantDoctor)
2. Make your changes
3. Open a pull request or submit via HF collaboration

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details *(coming soon)*

---

## 🔒 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file.

---

## ✅ Reproducibility Checklist (SoftwareX)

- [x] Open-source code & model weights included
- [x] Executable via Docker and Python
- [x] Zenodo DOI for archiving
- [x] Environment dependencies listed (`requirements.txt`)
- [x] Descriptive README with local setup, usage, and citation

---

## 🌐 Maintainer

[Rayean Patric](https://www.linkedin.com/in/rayeanpatric) • [rayeanpatric](https://github.com/rayeanpatric)

Have ideas or feedback? DM me on LinkedIn or open an issue!

---

Made with ❤️ by Rayean Patric

---
