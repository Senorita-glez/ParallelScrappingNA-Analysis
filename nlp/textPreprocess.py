
def limpiar_texto(texto):
    import re
    import spacy
    from nltk.corpus import stopwords
    from langdetect import detect, LangDetectException

    nlp_es = spacy.load('es_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')

    def obtener_stopwords(idioma):
        if idioma == 'es':
            stop_words = set(stopwords.words('spanish'))
        else:
            stop_words = set(stopwords.words('english'))
        stop_words.discard('no')  
        return stop_words

    if not texto.strip(): 
        return None  

    try:
        idioma = detect(texto)
    except LangDetectException:
        return texto

    nlp = nlp_es if idioma == 'es' else nlp_en
    doc = nlp(re.sub(r'[^\w\s]', '', texto.lower()))
    
    stop_words = obtener_stopwords(idioma)
    
    texto_limpio = ' '.join([
        token.lemma_ if token.text.lower() != 'no' else token.text  
        for token in doc if token.text.lower() not in stop_words and not token.is_digit
    ])

    return texto_limpio if texto_limpio.strip() else None


def preprocessText(name, listT, lock):
    for text in listT:
        newText = str(text)
        newText = limpiar_texto(newText)  # Limpiar el texto
        with lock:  
            with open(f'preprocess{name}Text.txt', 'a', encoding='utf-8') as f:
                f.write(newText + '\n')
