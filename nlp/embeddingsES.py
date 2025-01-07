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

data_espanol_balanceado = balancear_datos(data_espanol, 'sentimiento')

# Aplicar balanceo a los datos en inglés
data_ingles_balanceado = balancear_datos(data_ingles, 'sentimiento')

# Mostrar muestras de los DataFrames resultantes
'''print("Datos en español:")
display(data_espanol.head())

print("\nDatos en inglés:")
display(data_ingles.head())'''


# Cargar modelo de Spacy para preprocesamiento
nlp = spacy.load('es_core_news_sm')

# Definir stop words adicionales
STOP_WORDS_EXTRA = {"\u00a1", "-", "\u2014", "http", "<", ">"}
for word in STOP_WORDS_EXTRA:
    nlp.Defaults.stop_words.add(word)

# Función para procesar texto
def procesar_texto(texto):
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
                    tokens.append(num2words(numero, lang='es'))
                except ValueError:
                    pass
            else:
                lemma = token.text.lower()
                if lemma and lemma not in STOP_WORDS_EXTRA:
                    tokens.append(lemma)
        oraciones_tokenizadas.append(" ".join(tokens))

    return total_oraciones, oraciones_tokenizadas

# Función para preprocesar un subconjunto de datos
def procesar_textos_en_paralelo(data_slice_es):
    resultados = [procesar_texto(texto) for texto in data_slice_es['texto']]
    total_oraciones_es, oraciones_tokenizadas_es = zip(*resultados)
    return list(total_oraciones_es), list(oraciones_tokenizadas_es)

# Inicializar tokenizer y modelo BERT
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", ignore_mismatched_sizes=True)

# Función para obtener el embedding de una oración
def obtener_embedding_oracion(oracion):
    inputs = tokenizer(oracion, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Función para obtener el embedding promedio de una reseña
def obtener_embedding_resena(oraciones_tokenizadas_es):
    embeddings = [obtener_embedding_oracion(oracion) for oracion in oraciones_tokenizadas_es]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.config.hidden_size)

# Función para procesar un subconjunto de datos y generar embeddings
def procesar_rango(data_slice_es):
    return [obtener_embedding_resena(oraciones) for oraciones in data_slice_es['oraciones_tokenizadas_es']]

# Función para dividir los datos en subconjuntos
def dividir_datos(data_es, n_cores):
    longitud = len(data_es)
    p_mas = longitud % n_cores
    tamano_subgrupo = math.floor(longitud / n_cores)
    return [(i * tamano_subgrupo + min(i, p_mas), (i + 1) * tamano_subgrupo + min(i + 1, p_mas)) for i in range(n_cores)]

if __name__ == '__main__':
    # Número de núcleos para procesamiento paralelo
    n_cores = 7

    # Cargar los datos
    data_es = data_espanol

    # Dividir los datos para preprocesamiento
    limites_es = dividir_datos(data_es, n_cores)
    data_slices_es = [data_es.iloc[lim_inf:lim_sup] for lim_inf, lim_sup in limites_es]

    # Preprocesar textos en paralelo
    with Pool(n_cores) as pool:
        resultados_preprocesamiento_es = pool.map(procesar_textos_en_paralelo, data_slices_es)

    # Consolidar resultados del preprocesamiento
    total_oraciones_es = []
    oraciones_tokenizadas_es = []
    for total_es, tokenizadas_es in resultados_preprocesamiento_es:
        total_oraciones_es.extend(total_es)
        oraciones_tokenizadas_es.extend(tokenizadas_es)

    data_es['total_oraciones_es'] = total_oraciones_es
    data_es['oraciones_tokenizadas_es'] = oraciones_tokenizadas_es

    # Dividir los datos para generación de embeddings
    limites_es = dividir_datos(data_es, n_cores)
    data_slices_es = [data_es.iloc[lim_inf:lim_sup] for lim_inf, lim_sup in limites_es]

    # Generar embeddings en paralelo
    with Pool(n_cores) as pool:
        resultados_embeddings_es = pool.map(procesar_rango, data_slices_es)

    # Consolidar resultados de embeddings
    embeddings_es = [embedding for resultado in resultados_embeddings_es for embedding in resultado]
    data_es['embedding_resena_es'] = embeddings_es

    # Guardar embeddings con etiquetas
    embeddings_df_es = pd.DataFrame(data_es['embedding_resena_es'].to_list())
    embeddings_df_es['sentimiento'] = data_es['sentimiento'].values
    embeddings_df_es.to_csv('embeddings_con_etiqueta_es.csv', index=False)

    print("Embeddings por reseña con etiquetas guardados en 'embeddings_con_etiqueta_es.csv'.")
