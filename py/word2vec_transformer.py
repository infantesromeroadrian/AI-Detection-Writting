from gensim.models import Word2Vec
import numpy as np

class Word2VecTransformer:
    def __init__(self, dataframe, text_column, model=None):
        """
        Inicializa la clase Word2VecTransformer.

        :param dataframe: DataFrame a transformar.
        :param text_column: Nombre de la columna de texto.
        :param model: Modelo Word2Vec preentrenado o None para entrenar uno nuevo.
        """
        self.dataframe = dataframe
        self.text_column = text_column
        self.model = model

    def train_word2vec(self, vector_size=100, window=5, min_count=1, workers=4):
        """
        Entrena un modelo Word2Vec con los datos proporcionados.

        :param vector_size: Tamaño del vector de características.
        :param window: Máximo número de palabras entre la actual y la predicha dentro de una oración.
        :param min_count: Ignora palabras con frecuencia total menor a esto.
        :param workers: Hilos a utilizar en el entrenamiento.
        """
        tokenized_texts = self.dataframe[self.text_column].apply(lambda x: x.split())
        self.model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    def transform_texts_to_vectors(self):
        """
        Transforma los textos en vectores utilizando el modelo Word2Vec.
        """
        def text_to_vector(text):
            words = text.split()
            vectors = [self.model.wv[word] for word in words if word in self.model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.vector_size)

        self.dataframe['word2vec_vector'] = self.dataframe[self.text_column].apply(text_to_vector)