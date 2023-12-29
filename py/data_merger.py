
import pandas as pd
class DataMerger:
    def __init__(self, df1, df2):
        """
        Inicializa la clase DataMerger con dos DataFrames.

        :param df1: Primer DataFrame.
        :param df2: Segundo DataFrame.
        """
        self.df1 = df1
        self.df2 = df2
        self.merged_df = None

    def merge_data(self, key, how='inner'):
        """
        Fusiona los dos DataFrames usando una clave común.

        :param key: Columna clave para la fusión.
        :param how: Tipo de fusión - 'inner', 'outer', 'left', 'right' (default 'inner').
        """
        try:
            self.merged_df = pd.merge(self.df1, self.df2, on=key, how=how)
            print("Fusión realizada con éxito.")
        except Exception as e:
            print(f"Error al fusionar: {e}")

    def preview_merged_data(self, n=5):
        """
        Muestra las primeras n líneas del DataFrame fusionado.

        :param n: Número de líneas a mostrar (default 5).
        """
        if self.merged_df is not None:
            return self.merged_df.head(n)
        else:
            print("Los datos aún no han sido fusionados.")
            return None