# Importación de las clases
from data_loader import DataLoader
from data_inspector import DataInspector
from data_merger import DataMerger
from data_preprocessor import DataPreprocessor
from text_similarity import TextSimilarity
from data_visualizer import DataVisualizer
from word2vec_transformer import Word2VecTransformer
from data_splitter_balancer import DataSplitterBalancer
from text_data_classifier import TextClassifierModel
import pandas as pd

def main():
    # Cargar datos
    filepath_prompts = '/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/Kaggle/WrittingDetection/data/train_prompts.csv'
    filepath_essays = '/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/Kaggle/WrittingDetection/data/train_essays.csv'

    loader_prompts = DataLoader(filepath_prompts)
    loader_essays = DataLoader(filepath_essays)

    loader_prompts.load_data()
    loader_essays.load_data()

    # Inspeccionar datos
    inspector_prompts = DataInspector(loader_prompts.data)
    inspector_essays = DataInspector(loader_essays.data)

    inspector_prompts.get_info()
    inspector_essays.get_info()

    # Fusionar datos
    merger = DataMerger(loader_prompts.data, loader_essays.data)
    merger.merge_data(key='prompt_id')

    # Preprocesar datos
    preprocessor = DataPreprocessor(merger.merged_df)
    preprocessor.preprocess_data()

    # Calcular similitud de texto y visualización
    similarity = TextSimilarity(preprocessor.dataframe, 'cleaned_text')
    similarity.calculate_embeddings()
    similarity.add_max_cosine_similarity()

    visualizer = DataVisualizer(similarity.dataframe)
    visualizer.plot_class_balance('generated')
    visualizer.generate_wordcloud('cleaned_text')

    # Transformación con Word2Vec
    word2vec_transformer = Word2VecTransformer(similarity.dataframe, 'cleaned_text')
    word2vec_transformer.train_word2vec()
    word2vec_transformer.transform_texts_to_vectors()

    # Dividir y balancear datos
    splitter_balancer = DataSplitterBalancer(similarity.dataframe, 'generated', 'cleaned_text')
    train_df, val_df, test_df = splitter_balancer.split_data()
    balanced_train_df = splitter_balancer.balance_classes(train_df)

    # Inicializar y entrenar el modelo de clasificación de texto
    text_classifier = TextClassifierModel()
    text_classifier.train(balanced_train_df['cleaned_text'], balanced_train_df['generated'],
                          val_df['cleaned_text'], val_df['generated'])

    # Evaluación del modelo
    test_predictions = text_classifier.predict(test_df['cleaned_text'])
    # Aquí puedes agregar código para evaluar las predicciones, como calcular la exactitud, etc.

    # Realizar una predicción de prueba
    test_text = "Texto de ejemplo para clasificar"
    prediction = text_classifier.predict([test_text])
    print(f"Predicción: {prediction}")

if __name__ == "__main__":
    main()
