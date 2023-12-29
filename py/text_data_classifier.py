import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import L2

class TextClassifierModel:
    def __init__(self, max_vocab_size=10000, max_length=100, embedding_dim=32, lstm_units=64, num_classes=2):
        """
        Inicializa la clase TextClassifierModel.

        :param max_vocab_size: Tamaño máximo del vocabulario.
        :param max_length: Longitud máxima de las secuencias.
        :param embedding_dim: Dimensión del embedding.
        :param lstm_units: Unidades en la capa LSTM.
        :param num_classes: Número de clases para clasificación.
        """
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        self.model = self._build_model()

    def _build_model(self):
        """
        Construye el modelo de clasificación de texto.
        """
        model = Sequential([
            Embedding(self.max_vocab_size, self.embedding_dim, input_length=self.max_length),
            LSTM(self.lstm_units, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation='relu', kernel_regularizer=L2(0.001)),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, train_texts, train_labels, val_texts, val_labels, epochs=10, batch_size=32):
        """
        Entrena el modelo.

        :param train_texts: Textos de entrenamiento.
        :param train_labels: Etiquetas de entrenamiento.
        :param val_texts: Textos de validación.
        :param val_labels: Etiquetas de validación.
        :param epochs: Número de épocas para el entrenamiento.
        :param batch_size: Tamaño del lote.
        """
        self.tokenizer.fit_on_texts(train_texts)
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)
        train_padded = pad_sequences(train_sequences, maxlen=self.max_length, padding='post', truncating='post')

        val_sequences = self.tokenizer.texts_to_sequences(val_texts)
        val_padded = pad_sequences(val_sequences, maxlen=self.max_length, padding='post', truncating='post')

        self.model.fit(train_padded, train_labels, epochs=epochs, batch_size=batch_size,
                       validation_data=(val_padded, val_labels), verbose=2)

    def predict(self, texts):
        """
        Realiza predicciones con el modelo.

        :param texts: Textos para clasificar.
        :return: Predicciones del modelo.
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        return self.model.predict(padded)
