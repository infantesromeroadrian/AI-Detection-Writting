import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class DataSplitterBalancer:
    def __init__(self, dataframe, target_column, text_column):
        """
        Inicializa la clase DataSplitterBalancer.

        :param dataframe: DataFrame a dividir y balancear.
        :param target_column: Nombre de la columna objetivo para balancear.
        :param text_column: Nombre de la columna que contiene el texto.
        """
        self.dataframe = dataframe
        self.target_column = target_column
        self.text_column = text_column

    def split_data(self, test_size=0.2, val_size=0.1, random_state=None):
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba.

        :param test_size: Proporción del conjunto de prueba.
        :param val_size: Proporción del conjunto de validación.
        :param random_state: Semilla para la generación aleatoria.
        :return: Cuatro DataFrames: entrenamiento, validación, prueba y sus etiquetas.
        """
        # Filtrar solo las columnas relevantes
        relevant_data = self.dataframe[[self.text_column, self.target_column]]

        # Dividir en entrenamiento y prueba
        train_df, test_df = train_test_split(relevant_data, test_size=test_size,
                                             random_state=random_state, stratify=relevant_data[self.target_column])

        # Dividir el conjunto de entrenamiento en entrenamiento y validación
        train_df, val_df = train_test_split(train_df, test_size=val_size / (1 - test_size),
                                            random_state=random_state, stratify=train_df[self.target_column])

        return train_df, val_df, test_df

    def balance_classes(self, dataframe):
        """
        Balancea las clases en el DataFrame dado.

        :param dataframe: DataFrame a balancear.
        :return: DataFrame balanceado.
        """
        # Separar por clase
        df_majority = dataframe[dataframe[self.target_column] == dataframe[self.target_column].mode()[0]]
        df_minority = dataframe[dataframe[self.target_column] != dataframe[self.target_column].mode()[0]]

        # Balancear clases
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)

        # Combinar clases balanceadas
        df_balanced = pd.concat([df_majority, df_minority_upsampled])

        return df_balanced