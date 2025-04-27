# Contributing to PlantDoctor 🌱🧠

Thank you for considering contributing to **PlantDoctor** — an AI-powered plant disease diagnostic tool and agricultural chatbot. We welcome contributions of all kinds: bug fixes, features, model improvements, UI enhancements, or documentation updates.

---

## 📌 Getting Started

1. **Fork** the repository
2. **Clone** your fork locally:
   ```
   git clone https://github.com/your-username/PlantDoctor.git
   cd PlantDoctor
   ```
3. **Install dependencies** using `pip`:
   ```
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
   Or with conda:
   ```
   conda env create -f environment.yml
   conda activate plantdoctor
   ```

4. **Run locally** to test:
   ```
   python app.py
   ```

---

## 🛠️ Types of Contributions

- **🐛 Bug Fixes:** Identify issues and submit patches
- **✨ Features:** Add new diagnosis support, UI enhancements, chatbot upgrades
- **🧪 Testing:** Add test cases for prediction & image preprocessing
- **📝 Docs:** Improve installation steps, add FAQs
- **🚀 Performance:** Optimize model load time or chatbot latency

---

## 📂 Project Structure

```
app.py               # Gradio UI
chat_app.py          # Chatbot logic (Grok API)
model_loader.py      # ML model loading
utils.py             # Image processing + prediction
style.css            # Custom styles
attached_assets/     # Sample files
```

---

## ✅ Guidelines

- Follow [PEP8](https://peps.python.org/pep-0008/) coding style.
- Use clear commit messages:
  ```
  git commit -m "fix: handle None in predict_disease()"
  ```
- Test your changes before pushing.
- Open a **Pull Request (PR)** and describe your changes clearly.

---

## 💬 Community

Feel free to open:
- [Discussions](https://github.com/rayeanpatric/PlantDoctor/discussions)
- [Issues](https://github.com/rayeanpatric/PlantDoctor/issues)

We're happy to collaborate and improve this project together!

---
