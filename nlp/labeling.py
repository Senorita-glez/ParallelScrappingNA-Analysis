from multiprocessing import Lock
from collections import Counter
from joblib import load
import csv
import os

# Cargar los modelos entrenados para ambos idiomas
svm_model_es = load('nlp/sentimentAnalysis/modelsES/svm_model_es.joblib')
rn_model_es = load('nlp/sentimentAnalysis/modelsES/rn_model_es.joblib')
nb_model_es = load('nlp/sentimentAnalysis/modelsES/nb_model_es.joblib')

svm_model_en = load('nlp/sentimentAnalysis/modelsEN/svm_model_en.joblib')
rn_model_en = load('nlp/sentimentAnalysis/modelsEN/rn_model_en.joblib')
nb_model_en = load('nlp/sentimentAnalysis/modelsEN/nb_model_en.joblib')

def label_worker_row(row, lock, output_file):
    """
    Clasifica una fila de (id, idioma, embedding) y guarda la predicción en un archivo.

    Args:
        row (tuple): Una fila con (id, idioma, embedding).
        lock (Lock): Lock para asegurar escritura concurrente.
        output_file (str): Ruta del archivo donde se guardarán las predicciones.
    """
    import numpy as np

    id_, idioma, embedding = row

    # Convertir embedding a NumPy array y verificar dimensiones
    embedding = np.array(embedding).reshape(1, -1)  # Asegurar que sea 2D

    # Seleccionar modelos según el idioma
    if idioma == 'es':
        pred_svm = svm_model_es.predict(embedding)[0]
        pred_rn = rn_model_es.predict(embedding)[0]
        pred_nb = nb_model_es.predict(embedding)[0]
    elif idioma == 'en':
        pred_svm = svm_model_en.predict(embedding)[0]
        pred_rn = rn_model_en.predict(embedding)[0]
        pred_nb = nb_model_en.predict(embedding)[0]
    else:
        # Si el idioma es desconocido, asignar "unknown"
        final_prediction = "unknown"
    if idioma in ['es', 'en']:
        # Combinar predicciones mediante votación mayoritaria
        final_prediction = Counter([pred_svm, pred_rn, pred_nb]).most_common(1)[0][0]

    # Asegurar escritura segura en el archivo
    with lock:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Escribir en el archivo
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([id_, idioma, final_prediction])

        
def label_worker_batch(rows, lock, output_file):
    """
    Clasifica un lote de filas y guarda las predicciones en un archivo.

    Args:
        rows (list): Lista de filas (id, idioma, embedding).
        lock (Lock): Lock para asegurar escritura concurrente.
        output_file (str): Ruta del archivo donde se guardarán las predicciones.
    """
    for row in rows:
        label_worker_row(row, lock, output_file)