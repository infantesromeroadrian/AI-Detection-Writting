import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        """
        Inicializa la clase DataLoader con la ruta del archivo de datos.

        :param filepath: Ruta del archivo de datos.
        """
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """
        Carga los datos desde el archivo especificado.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            print("Datos cargados correctamente.")
        except Exception as e:
            print(f"Error al cargar los datos: {e}")

    def preview_data(self, n=5):
        """
        Muestra las primeras n líneas de los datos cargados.

        :param n: Número de líneas a mostrar (default 5).
        """
        if self.data is not None:
            return self.data.head(n)
        else:
            print("Los datos aún no han sido cargados.")
            return None