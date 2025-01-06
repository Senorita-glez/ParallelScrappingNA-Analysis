import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException

# Cargar los modelos de spaCy para español e inglés
nlp_es = spacy.load('es_core_news_sm')
nlp_en = spacy.load('en_core_web_sm')

# Función para obtener stopwords según el idioma
def obtener_stopwords(idioma):
    if idioma == 'es':
        stop_words = set(stopwords.words('spanish'))
    else:
        stop_words = set(stopwords.words('english'))
    stop_words.discard('no')  # Mantener "no" sin eliminar
    return stop_words

# Función para limpiar el texto (todo en una sola función)
def limpiar_texto(texto):
    import re
    import spacy
    from nltk.corpus import stopwords
    from langdetect import detect, LangDetectException

    # Cargar los modelos de spaCy para español e inglés
    nlp_es = spacy.load('es_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')

    # Función para obtener stopwords según el idioma
    def obtener_stopwords(idioma):
        if idioma == 'es':
            stop_words = set(stopwords.words('spanish'))
        else:
            stop_words = set(stopwords.words('english'))
        stop_words.discard('no')  # Mantener "no" sin eliminar
        return stop_words

    if not texto.strip():  # Si el texto está vacío o solo tiene espacios
        return None  # Regresar None para eliminar los vacíos en el DataFrame

    try:
        # Detectar el idioma del texto
        idioma = detect(texto)
    except LangDetectException:
        return texto

    # Seleccionar el modelo spaCy adecuado según el idioma detectado
    nlp = nlp_es if idioma == 'es' else nlp_en
    # Limpiar el texto: eliminar puntuación, convertir a minúsculas
    doc = nlp(re.sub(r'[^\w\s]', '', texto.lower()))
    
    # Obtener las stopwords del idioma
    stop_words = obtener_stopwords(idioma)
    
    # Lematizar y filtrar palabras vacías y números
    texto_limpio = ' '.join([
        token.lemma_ if token.text.lower() != 'no' else token.text  # Mantener "no" sin lematizar
        for token in doc if token.text.lower() not in stop_words and not token.is_digit
    ])
    
    # Si el texto limpio tiene solo espacios (es decir, está vacío después de la limpieza), retornar None
    return texto_limpio if texto_limpio.strip() else None

# Cargar los comentarios en un DataFrame
with open('reviews_dataAmazon.csv', 'r', encoding='utf-8') as file:
    comentarios = file.readlines()

df = pd.DataFrame(comentarios, columns=['comentario'])

# Aplicar la función de limpieza
df['comentario_limpio'] = df['comentario'].apply(limpiar_texto)

# Eliminar filas con comentarios vacíos (None)
df = df[df['comentario_limpio'].notna()]

# Guardar el DataFrame limpio en un nuevo archivo CSV
df_limpio = df[['comentario_limpio']]
df_limpio.to_csv('comentarios_limpios.csv', index=False, encoding='utf-8')
