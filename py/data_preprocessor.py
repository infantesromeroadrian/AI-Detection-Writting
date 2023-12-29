import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string

# Descargando los recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataPreprocessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def preprocess_data(self):
        """
        Realiza el preprocesamiento de los datos fusionando las columnas específicas en una
        y luego aplicando técnicas avanzadas de preprocesamiento de texto.
        """
        # Fusionando las columnas en una nueva columna 'combined_text'
        self.dataframe['combined_text'] = (self.dataframe['prompt_name'] + " " +
                                           self.dataframe['instructions'] + " " +
                                           self.dataframe['source_text'] + " " +
                                           self.dataframe['text'])

        # Aplicando preprocesamiento avanzado
        self.dataframe['cleaned_text'] = self.dataframe['combined_text'].apply(self.clean_text_advanced)

        # Eliminando las columnas originales que ya no son necesarias
        columns_to_drop = ['prompt_name', 'instructions', 'source_text', 'text']
        self.dataframe.drop(columns=columns_to_drop, axis=1, inplace=True)

    def clean_text_advanced(self, text):
        """
        Realiza la limpieza avanzada del texto, incluyendo tokenización, eliminación de stop words,
        lematización y otros procesos de limpieza.
        """
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar puntuación
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenización
        tokens = word_tokenize(text)
        # Eliminación de stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lematización
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
        # Uniendo los tokens limpios en un solo string
        cleaned_text = " ".join([word for word in lemmatized if word.isalpha()])
        return cleaned_text