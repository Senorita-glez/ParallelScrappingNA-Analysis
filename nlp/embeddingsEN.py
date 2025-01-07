import pandas as pd
import numpy as np
from num2words import num2words
import spacy
from transformers import AutoTokenizer, AutoModel
from multiprocessing import Pool
import math
import torch
import pandas as pd
from sklearn.utils import resample

# Balancear datos mediante sobremuestreo
def balancear_datos(df, clase_objetivo):
    """
    Balancea un DataFrame duplicando muestras de la clase minoritaria.

    Args:
        df (DataFrame): DataFrame a balancear.
        clase_objetivo (str): Nombre de la columna con las etiquetas.

    Returns:
        DataFrame: DataFrame balanceado.
    """
    # Separar clases mayoritaria y minoritaria
    clase_mayoritaria = df[df[clase_objetivo] == '1']
    clase_minoritaria = df[df[clase_objetivo] == '0']

    # Sobremuestrear clase minoritaria
    clase_minoritaria_balanceada = resample(
        clase_minoritaria,
        replace=True,  # Permitir duplicados
        n_samples=len(clase_mayoritaria),  # Igualar número de muestras
        random_state=42
    )

    # Combinar clases y devolver el resultado balanceado
    return pd.concat([clase_mayoritaria, clase_minoritaria_balanceada])

# Cargar datos
url = "https://raw.githubusercontent.com/ignaciomsarmiento/RecomSystemsLectures/main/L07_sentimientos/data/Amazon.csv"
data = pd.read_csv(url)

# Verificar columnas necesarias
required_columns = ['reviews.text', 'reviews.text_esp', 'reviews.rating']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Las siguientes columnas necesarias están ausentes en los datos: {missing_columns}")

# Seleccionar columnas necesarias
data = data[['reviews.text', 'reviews.text_esp', 'reviews.rating']].copy()

# Asignar sentimiento
data['sentimiento'] = data['reviews.rating'].apply(lambda x: '1' if x >= 4.0 else '0')

# Separar datos en dos DataFrames, uno para español y otro para inglés
data_espanol = data[['reviews.text_esp', 'sentimiento']].rename(columns={'reviews.text_esp': 'texto'})
data_ingles = data[['reviews.text', 'sentimiento']].rename(columns={'reviews.text': 'texto'})

# Aplicar balanceo a los datos en español
data_espanol_balanceado = balancear_datos(data_espanol, 'sentimiento')

# Aplicar balanceo a los datos en inglés
data_ingles_balanceado = balancear_datos(data_ingles, 'sentimiento')

# Cargar modelo de Spacy para preprocesamiento
nlp = spacy.load('en_core_web_sm')

# Definir stop words adicionales
STOP_WORDS_EXTRA = {"\u00a1", "-", "\u2014", "http", "<", ">"}
for word in STOP_WORDS_EXTRA:
    nlp.Defaults.stop_words.add(word)

# Función para procesar texto
def procesar_texto_en(texto):
    """
    Procesa un texto dividiéndolo en oraciones y tokenizándolas.
    Convierte números a palabras y elimina stop words.
    """
    oraciones = texto.split('\n')
    total_oraciones = len(oraciones)
    oraciones_tokenizadas = []

    for oracion in oraciones:
        tokens = []
        doc = nlp(oracion)
        for token in doc:
            if token.is_digit:
                try:
                    numero = int(token.text)
                    tokens.append(num2words(numero, lang='en'))
                except ValueError:
                    pass
            else:
                lemma = token.text.lower()
                if lemma and lemma not in STOP_WORDS_EXTRA:
                    tokens.append(lemma)
        oraciones_tokenizadas.append(" ".join(tokens))

    return total_oraciones, oraciones_tokenizadas

# Función para preprocesar un subconjunto de datos
def procesar_textos_en_paralelo_en(data_slice_en):
    resultados = [procesar_texto_en(texto) for texto in data_slice_en['texto']]
    total_oraciones_en, oraciones_tokenizadas_en = zip(*resultados)
    return list(total_oraciones_en), list(oraciones_tokenizadas_en)

# Inicializar tokenizer y modelo BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", ignore_mismatched_sizes=True)

# Función para obtener el embedding de una oración
def obtener_embedding_oracion_en(oracion):
    inputs = tokenizer(oracion, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Función para obtener el embedding promedio de una reseña
def obtener_embedding_resena_en(oraciones_tokenizadas_en):
    embeddings = [obtener_embedding_oracion_en(oracion) for oracion in oraciones_tokenizadas_en]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.config.hidden_size)

# Función para procesar un subconjunto de datos y generar embeddings
def procesar_rango_en(data_slice_en):
    return [obtener_embedding_resena_en(oraciones) for oraciones in data_slice_en['oraciones_tokenizadas_en']]

# Función para dividir los datos en subconjuntos
def dividir_datos_en(data_en, n_cores):
    longitud = len(data_en)
    p_mas = longitud % n_cores
    tamano_subgrupo = math.floor(longitud / n_cores)
    return [(i * tamano_subgrupo + min(i, p_mas), (i + 1) * tamano_subgrupo + min(i + 1, p_mas)) for i in range(n_cores)]

if __name__ == '__main__':
    # Número de núcleos para procesamiento paralelo
    n_cores = 10

    # Cargar los datos
    data_en = data_ingles  # Reemplazar con la ruta correcta

    # Dividir los datos para preprocesamiento
    limites_en = dividir_datos_en(data_en, n_cores)
    data_slices_en = [data_en.iloc[lim_inf:lim_sup] for lim_inf, lim_sup in limites_en]

    # Preprocesar textos en paralelo
    with Pool(n_cores) as pool:
        resultados_preprocesamiento_en = pool.map(procesar_textos_en_paralelo_en, data_slices_en)

    # Consolidar resultados del preprocesamiento
    total_oraciones_en = []
    oraciones_tokenizadas_en = []
    for total_en, tokenizadas_en in resultados_preprocesamiento_en:
        total_oraciones_en.extend(total_en)
        oraciones_tokenizadas_en.extend(tokenizadas_en)

    data_en['total_oraciones_en'] = total_oraciones_en
    data_en['oraciones_tokenizadas_en'] = oraciones_tokenizadas_en

    # Dividir los datos para generación de embeddings
    limites_en = dividir_datos_en(data_en, n_cores)
    data_slices_en = [data_en.iloc[lim_inf:lim_sup] for lim_inf, lim_sup in limites_en]

    # Generar embeddings en paralelo
    with Pool(n_cores) as pool:
        resultados_embeddings_en = pool.map(procesar_rango_en, data_slices_en)

    # Consolidar resultados de embeddings
    embeddings_en = [embedding for resultado in resultados_embeddings_en for embedding in resultado]
    data_en['embedding_resena_en'] = embeddings_en

    # Guardar embeddings con etiquetas
    embeddings_df_en = pd.DataFrame(data_en['embedding_resena_en'].to_list())
    embeddings_df_en['sentimiento'] = data_en['sentimiento'].values
    embeddings_df_en.to_csv('embeddings_con_etiqueta_en.csv', index=False)

    print("Embeddings por reseña con etiquetas guardados en 'embeddings_con_etiqueta_en.csv'.")
