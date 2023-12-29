class DataInspector:
    def __init__(self, dataframe):
        """
        Inicializa la clase DataInspector con un DataFrame de pandas.

        :param dataframe: DataFrame de pandas a inspeccionar.
        """
        self.dataframe = dataframe

    def get_info(self):
        """
        Imprime información general del DataFrame, incluyendo tipos de datos y valores no nulos.
        """
        return self.dataframe.info()

    def get_description(self):
        """
        Devuelve un resumen estadístico del DataFrame para las columnas numéricas.
        """
        return self.dataframe.describe()

    def check_nulls(self):
        """
        Verifica y cuenta los valores nulos en el DataFrame.
        """
        return self.dataframe.isnull().sum()