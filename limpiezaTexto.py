import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException

nlp_es = spacy.load('es_core_news_sm')
nlp_en = spacy.load('en_core_web_sm')

with open('../reviews_dataAmazon.csv', 'r', encoding='utf-8') as file:  #aguas con el archivo, el que puse solo es de prueba
    comentarios = file.readlines()

df = pd.DataFrame(comentarios, columns=['comentario'])

def obtener_stopwords(idioma):
    if idioma == 'es':
        stop_words = set(stopwords.words('spanish'))
    else:
        stop_words = set(stopwords.words('english'))
    stop_words.discard('no')  # Mantener "no"
    return stop_words

def limpiar_texto(texto):
    if not texto.strip():
        return ""
    try:
        idioma = detect(texto)
    except LangDetectException:
        return texto

    nlp = nlp_es if idioma == 'es' else nlp_en
    doc = nlp(re.sub(r'[^\w\s]', '', texto.lower()))  # Limpiar texto 
    stop_words = obtener_stopwords(idioma)
    
    return ' '.join([
        token.lemma_ if token.text.lower() != 'no' else token.text  # Mantener "no" sin lematizar
        for token in doc if token.text.lower() not in stop_words and not token.is_digit
    ])

df['comentario_limpio'] = df['comentario'].apply(limpiar_texto)
df_limpio = df[['comentario_limpio']]
df_limpio.to_csv('comentarios_limpios.csv', index=False, encoding='utf-8') # se crea otro archivo nuevo, en caso de aruinar el original
