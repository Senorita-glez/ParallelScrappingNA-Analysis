# Función para obtener stopwords
def obtener_stopwords(idioma):
    import nltk
    from nltk.corpus import stopwords  # Import stopwords module
    
    # Get stopwords based on the specified language
    if idioma == 'es':
        stop_words = set(stopwords.words('spanish'))
    else:
        stop_words = set(stopwords.words('english'))
    
    stop_words.discard('no')  # Remove 'no' from the set of stopwords
    return stop_words



def limpiar_texto(texto):
    
    from langdetect import detect, LangDetectException
    import re 
    import spacy
    nlp_es = spacy.load('es_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')
    if not texto.strip():
        return None

    try:
        idioma = detect(texto)
    except LangDetectException:
        return texto

    texto = re.sub(r'[\n\t]', ' ', texto)  # Reemplazar saltos de línea y tabulaciones por espacios
    texto = re.sub(r'[^\w\s]', '', texto.lower())  # Eliminar caracteres no alfanuméricos excepto espacios

    nlp = nlp_es if idioma == 'es' else nlp_en
    doc = nlp(texto)

    stop_words = obtener_stopwords(idioma)

    texto_limpio = ' '.join([
        token.lemma_ if token.text.lower() != 'no' else token.text  
        for token in doc if token.text.lower() not in stop_words and not token.is_digit
    ])

    return texto_limpio if texto_limpio.strip() else None

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", ignore_mismatched_sizes=True)

def obtener_embedding_oracion(oracion):
    import torch
    inputs = tokenizer(oracion, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def obtener_embedding_texto(texto):
    import numpy as np 
    texto_limpio = limpiar_texto(texto)
    if not texto_limpio:
        return np.zeros(model.config.hidden_size)  # Retornar un vector cero si el texto está vacío

    oraciones = texto_limpio.split('. ')  # Dividir en oraciones simples
    embeddings = [obtener_embedding_oracion(oracion) for oracion in oraciones if oracion.strip()]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.config.hidden_size)



def preprocessText(name, listT, lock):
    import csv
    import os
    from langdetect import detect, LangDetectException

    output_file = f'nlp/preprocess_{name}_Text.csv'
    os.makedirs('nlp', exist_ok=True)  # Asegúrate de que el directorio exista

    # Escribir encabezado solo si el archivo no existe
    with lock:
        if not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'idioma'] + [f'embedding_{i}' for i in range(model.config.hidden_size)])  # Encabezados

    # Procesar datos
    for review_id, text in listT:
        # Detectar el idioma
        try:
            idioma = detect(text)
        except LangDetectException:
            idioma = "unknown"  # Si no se puede detectar el idioma

        # Obtener el embedding del texto
        embedding = obtener_embedding_texto(text)
        if embedding is not None:
            with lock:
                with open(output_file, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([review_id, idioma] + embedding.tolist())  # Escribir id, idioma y embedding como lista
