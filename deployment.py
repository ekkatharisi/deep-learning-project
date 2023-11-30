import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf


def load_model(model_path):
    model = tf.lite.Interpreter(model_path=model_path)
    model.allocate_tensors()
    return model


def predict_image(img):
    model = load_model("./saved_final_model.tflite")
    img = img.resize((224, 224))

    img_array =  np.array(img).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Préparer les données pour le modèle
    input_tensor_index = model.get_input_details()[0]['index']
    output = model.tensor(model.get_output_details()[0]['index'])

    # Faire une prédiction
    model.set_tensor(input_tensor_index, img_array)
    model.invoke()
    prediction = output()

    if prediction[0, 0] > 0.5:
        return 'Dog'
    else:
        return 'Cat'


st.title("Image Classifier - Cat or Dog")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    result = predict_image(image)
    st.success(f"the predicted animal is : {result}")

