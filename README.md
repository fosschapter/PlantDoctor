# Plant Disease Diagnosis and Treatment

This application uses a MobileNetV2 deep learning model to diagnose plant diseases from leaf images and provides treatment recommendations along with a chatbot for agricultural queries.

## Features

- **Image-based Disease Diagnosis**: Upload leaf images to identify common plant diseases
- **Treatment Recommendations**: Get specific treatment advice for identified diseases
- **Agricultural Chatbot**: Ask questions about plant diseases, farming practices, and agricultural topics
- **Multiple Deployment Options**: Choose between Gradio or Streamlit interfaces

## Models and Data

- **Image Classification**: MobileNetV2 model trained on the PlantVillage dataset
- **Diseases Covered**: Various common plant diseases including tomato blights, apple scab, potato diseases, and more

## How to Use

1. **Diagnosis Feature**:
   - Upload a clear image of a plant leaf showing disease symptoms
   - The application will analyze and identify the disease
   - View the diagnosis result and treatment recommendations

2. **Agricultural Chatbot**:
   - Type questions about plant diseases, treatments, or general agriculture
   - Get informative responses based on agricultural knowledge

## Deployment Options

This application can be deployed in multiple ways to accommodate different hosting environments:

### Hugging Face Spaces

For deployment on Hugging Face Spaces, you can use either Gradio or Streamlit:

#### Option 1: Using Gradio (Recommended)

1. Create a new Space in Hugging Face
2. Select the **Gradio** SDK
3. Upload all files from this repository
4. In your Space settings:
   - Set `app.py` as the entry point file, or rename `app_gradio.py` to `app.py`
   - Use `requirements-huggingface.txt` for package installation
   - Make sure to include the model file `attached_assets/mobilenetv2.h5`

#### Option 2: Using Streamlit

1. Create a new Space in Hugging Face
2. Select the **Streamlit** SDK
3. Upload all files from this repository
4. In your Space settings:
   - Rename `app_streamlit.py` to `app.py` to serve as the entry point
   - Use `requirements-huggingface.txt` for package installation
   - Make sure to include the model file `attached_assets/mobilenetv2.h5`

#### Option 3: Universal Deployment

You can also use the unified entry point which supports both Gradio and Streamlit:

1. Create a new Space in Hugging Face
2. Select the **Gradio** SDK (default)
3. Upload all files from this repository
4. Rename `app_huggingface.py` to `app.py`
5. By default, this will use the Gradio interface
6. To use Streamlit instead, set the environment variable `INTERFACE=streamlit` in your Space settings

### Local Deployment

For local deployment, you can run either:

```bash
# For Gradio interface
python app_gradio.py

# For Streamlit interface
streamlit run app_streamlit.py

# For Flask interface (original)
python main.py
```

## Dependencies

- TensorFlow/Keras (for the MobileNetV2 model)
- Pillow (for image processing)
- NumPy (for numerical operations)
- Gradio or Streamlit (for the interface)
- Flask (for the original web interface)

## Technical Notes

- The model is a MobileNetV2 architecture trained on plant disease images
- The application includes a simplified chatbot that responds based on keyword matching
- Model file: `attached_assets/mobilenetv2.h5`
- Class labels: `class_labels.json`