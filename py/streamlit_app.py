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
st.title('Writting AI Detector App')

# Opción para subir archivo o ingresar texto
text_input_method = st.radio("How would you like to enter a text?", ('Write text', 'Upload file'))

text = ""
if text_input_method == 'Write text':
    text = st.text_area("Enter Text","Type Here ..")
elif text_input_method == 'Upload file':
    uploaded_file = st.file_uploader("Upload Files",type=['txt'])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

# Botón de clasificación
if st.button('Classify'):
    if text:
        result = classify_text(text)
        st.write("The text is: ")
        st.write(f"The text is {'AI Generated' if result == 1 else 'Human Writting'}")
    else:
        st.write("Please enter a text to classify")

# Ejecutar: streamlit run app.py


