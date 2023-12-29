import plotly.express as px
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from io import BytesIO
import base64
import urllib

class DataVisualizer:
    def __init__(self, dataframe):
        """
        Inicializa la clase DataVisualizer con un DataFrame.

        :param dataframe: DataFrame a visualizar.
        """
        self.dataframe = dataframe

    def plot_class_balance(self, column):
        """
        Visualiza el balance de clases de una columna específica.

        :param column: Nombre de la columna a visualizar.
        """
        count_df = self.dataframe[column].value_counts().reset_index()
        count_df.columns = ['Clase', 'Frecuencia']
        fig = px.bar(count_df, x='Clase', y='Frecuencia',
                     labels={'Clase': 'Clases', 'Frecuencia': 'Frecuencia'},
                     title=f'Balance de clases para la columna {column}')
        fig.show()

    def plot_lexical_richness(self, text_column):
        """
        Visualiza la riqueza léxica en una columna de texto.

        :param text_column: Nombre de la columna de texto.
        """
        text = ' '.join(self.dataframe[text_column].dropna())
        tokens = word_tokenize(text)
        vocab = set(tokens)
        lexical_richness = len(vocab) / len(tokens) if tokens else 0
        fig = px.bar(x=['Riqueza Léxica'], y=[lexical_richness],
                     title=f'Riqueza Léxica en {text_column}')
        fig.show()

    def generate_wordcloud(self, text_column):
        """
        Genera y muestra un WordCloud para una columna de texto específica.

        :param text_column: Nombre de la columna de texto.
        """
        text = ' '.join(self.dataframe[text_column].dropna())
        wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(text)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)
        fig = px.imshow(np.array(wordcloud.to_image()))
        fig.update_layout(title_text=f'WordCloud para {text_column}')
        fig.show()