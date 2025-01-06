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

# Función para limpiar texto
def limpiar_texto(texto):
    from langdetect import detect, LangDetectException
    import re
    import spacy

    # Cargar modelos de spaCy
    nlp_es = spacy.load('es_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')

    if not texto.strip(): 
        return None  

    try:
        # Detectar el idioma del texto
        idioma = detect(texto)
    except LangDetectException:
        return texto

    # Eliminar saltos de línea, tabulaciones y caracteres especiales no deseados
    texto = re.sub(r'[\n\t]', ' ', texto)  # Reemplazar saltos de línea y tabulaciones por un espacio
    texto = re.sub(r'[^\w\s]', '', texto.lower())  # Eliminar caracteres no alfanuméricos excepto espacios

    # Seleccionar el modelo adecuado según el idioma detectado
    nlp = nlp_es if idioma == 'es' else nlp_en
    doc = nlp(texto)
    
    # Obtener stopwords según el idioma
    stop_words = obtener_stopwords(idioma)
    
    # Procesar el texto, lematizar y eliminar palabras irrelevantes
    texto_limpio = ' '.join([
        token.lemma_ if token.text.lower() != 'no' else token.text  
        for token in doc if token.text.lower() not in stop_words and not token.is_digit
    ])

    return texto_limpio if texto_limpio.strip() else None

# Función para preprocesar texto
def preprocessText(name, listT, lock):
    import csv

    output_file = f'outputs/preprocess_{name}_Text.csv'
    # Escribir encabezado solo si el archivo no existe
    with lock:
        with open(output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # Si el archivo está vacío
                writer.writerow(['id', 'comentario_limpio'])  # Encabezados
    # Procesar datos
    for review_id, text in listT:
        cleaned_text = limpiar_texto(text)  # Limpiar el texto
        if cleaned_text is not None:
            with lock:
                with open(output_file, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([review_id, cleaned_text])