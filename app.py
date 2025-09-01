import os
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224,224)
DEFAULT_MODEL_DIR = 'saved_models'

def load_model_or_raise(model_path=None):
    if model_path and os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    #latest = os.path.join(DEFAULT_MODEL_DIR,'brain_tumor_mobilenetv2_latest.h5')
    latest = os.path.join(DEFAULT_MODEL_DIR,'brain_tumor_mobilenetv2_20250901_160237.h5')
    if os.path.exists(latest):
        return tf.keras.models.load_model(latest)
    raise FileNotFoundError('No model found. Train first or provide --model_path')

def prepare_image(img: Image.Image, target_size=IMG_SIZE):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def run_app(model_path=None):
    import streamlit as st
    st.set_page_config(page_title='TumorVision', layout='centered')
    st.title('Brain Tumor Detector')
    st.write('Upload MRI/brain image to detect tumor')

    try:
        model = load_model_or_raise(model_path)
        st.success('Model loaded successfully')
    except Exception as e:
        st.error(f'Could not load model: {e}')
        st.stop()

    uploaded_file = st.file_uploader('Upload image', type=['jpg','jpeg','png','bmp'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        x = prepare_image(image)
        pred = float(model.predict(x)[0][0])
        label = 'Tumor' if pred>=0.5 else 'No Tumor'
        if label == 'Tumor':
            st.subheader(f'🟥 {label}')
        else:
            st.subheader(f'🟩 {label}')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None)
    args = parser.parse_args()
    run_app(model_path=args.model_path)