import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Cargar el modelo y el tokenizer
model = load_model('/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/Kaggle/WrittingDetection/model/text_classifier_model.h5')
with open('/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/Kaggle/WrittingDetection/model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def classify_text(text):
    # Preprocesamiento del texto
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

    # Predicción
    prediction = model.predict(padded)
    predicted_class = prediction[0].argmax()
    return predicted_class

# Interfaz de Streamlit
st.title('Detector de Texto Generado por IA')

# Opción para subir archivo o ingresar texto
text_input_method = st.radio("¿Cómo deseas ingresar el texto?", ('Escribir texto', 'Subir archivo'))

text = ""
if text_input_method == 'Escribir texto':
    text = st.text_area("Ingresa el texto aquí:", height=200)
elif text_input_method == 'Subir archivo':
    uploaded_file = st.file_uploader("Elige un archivo de texto", type=['txt'])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

# Botón de clasificación
if st.button('Clasificar'):
    if text:
        result = classify_text(text)
        st.write("Resultado en bruto de la clasificación:", result)
        st.write(f"El texto es {'generado por IA' if result == 1 else 'escrito por un humano'}")
    else:
        st.write("Por favor, ingresa un texto o sube un archivo.")

# Ejecutar: streamlit run app.py


