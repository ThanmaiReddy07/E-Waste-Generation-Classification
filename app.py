import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the saved model
model = tf.keras.models.load_model("Efficient_classify.keras")

# Define class labels
class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

# Define prediction function
def classify_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    index = np.argmax(predictions)
    confidence = predictions[0][index]

    return f"Predicted: {class_names[index]} (Confidence: {confidence:.2f})"

# Build Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="E-Waste Image Classifier 🌍",
    description="Upload an image of an e-waste item and let the model predict its category.",
)

# For Render, you must use dynamic PORT and set server_name to "0.0.0.0"
import os
iface.launch(server_port=int(os.environ.get('PORT', 7860)), server_name="0.0.0.0")
