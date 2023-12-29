from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class TextSimilarity:
    def __init__(self, dataframe, text_column):
        """
        Inicializa la clase TextSimilarity con un DataFrame y el nombre de la columna de texto.

        :param dataframe: DataFrame que contiene los datos.
        :param text_column: Nombre de la columna que contiene el texto.
        """
        self.dataframe = dataframe
        self.text_column = text_column
        self.tfidf_matrix = None
        self.vectorizer = TfidfVectorizer()

    def calculate_embeddings(self):
        """
        Calcula los embeddings TF-IDF para el texto y los almacena en el DataFrame.
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(self.dataframe[self.text_column])
        # Convertir cada fila de la matriz TF-IDF a una lista y almacenarla en el DataFrame
        self.dataframe['embeddings'] = self.tfidf_matrix.toarray().tolist()

    def add_max_cosine_similarity(self):
        """
        Calcula y agrega la similitud del coseno máxima de cada texto con todos los demás
        como una nueva columna en el DataFrame.
        """
        if self.tfidf_matrix is not None:
            cosine_sim = cosine_similarity(self.tfidf_matrix)
            # Excluyendo la similitud del texto consigo mismo (diagonal) y tomando el máximo
            max_sim = cosine_sim - np.eye(cosine_sim.shape[0])
            self.dataframe['max_cosine_similarity'] = max_sim.max(axis=1)
        else:
            print("Primero debe calcular los embeddings.")