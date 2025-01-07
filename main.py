if __name__ == '__main__':
    from scrapping.amazon import *
    from scrapping.mercadoLibre import *
    from utils.paralell import nivelacion_cargas
    from multiprocessing import Lock, Process
    from nlp.textProcess import preprocessText
    from utils.csvOperation import *
    from nlp.labeling import label_worker_batch
    import time 
    import pandas as pd
    import nltk
    import csv

    N_THREADS = 4
    lock = Lock()

    '''
    threadsScrapping = []

    product_name = input("\nProducto: ")
    linksML = getLinksML(product_name)
    linkscleanML = limpiar(linksML)
    linksAmazon = getLinksAmazon(product_name)

    sublistsML = nivelacion_cargas(linkscleanML, N_THREADS)
    sublistsAmazon = nivelacion_cargas(linksAmazon, N_THREADS)

    for i in range(N_THREADS):
        # Initialize each process for Mercado Libre and Amazon reviews
        threadsScrapping.append(Process(target=scrape_url, args=(sublistsML[i], lock)))
        threadsScrapping.append(Process(target=AmazonReviews, args=(sublistsAmazon[i], lock)))

    start_time = time.perf_counter()

    # Start all threads
    for threadSrc in threadsScrapping:
        threadSrc.start()

    # Wait for all threads to complete
    for threadScr in threadsScrapping:
        threadSrc.join()

    finish_time = time.perf_counter()
    print(f"Scrapping finished in {finish_time - start_time} seconds")
    '''

    # En esta parte se implementa la clasificación de modelos con los reviews obtenidos.

    threadsProcess = []

    pathReviewsAmazon = 'scrapping/outputScrapping/reviewsAmazon.csv'
    pathReviewsML = 'scrapping/outputScrapping/reviewsML.csv'

    '''addIdCsv(pathReviewsAmazon, 'comentarios')
    addIdCsv(pathReviewsML, 'comentarios')'''

    # Leer datos
    dfAmazon = pd.read_csv(pathReviewsAmazon)
    dfML = pd.read_csv(pathReviewsML)

    # Convertir a listas de tuplas (id, comentario)
    amazon_reviews = list(zip(dfAmazon['id'], dfAmazon['comentarios']))
    ml_reviews = list(zip(dfML['id'], dfML['comentarios']))

    # Dividir en sublistas para procesamiento paralelo
    subReviewsAmazon = nivelacion_cargas(amazon_reviews, N_THREADS)
    subReviewsML = nivelacion_cargas(ml_reviews, N_THREADS)

    nltk.download('stopwords')

    for i in range(N_THREADS):
        # Initialize each process for Mercado Libre and Amazon reviews
        threadsProcess.append(Process(target=preprocessText, args=('ML', subReviewsML[i], lock)))
        threadsProcess.append(Process(target=preprocessText, args=('Amazon', subReviewsAmazon[i], lock)))

    start_time = time.perf_counter()

    # Start all threads
    for threadPr in threadsProcess:
        threadPr.start()

    # Wait for all threads to complete
    for threadPr in threadsProcess:
        threadPr.join()

    finish_time = time.perf_counter()
    print(f"Text Process finished in {finish_time - start_time} seconds")


    pathEmmAmazon = 'nlp/preprocess_Amazon_Text.csv'
    pathEmmML = 'nlp/preprocess_ML_Text.csv'
    outputLabelsAmazon = 'nlp/sentimentAnalysis/labels_Amazon.csv'
    outputLabelsML = 'nlp/sentimentAnalysis/labels_ML.csv'

    # Leer datos de preprocesamiento
    print("Leyendo datos de Amazon y MercadoLibre...")
    dfAmazon = pd.read_csv(pathEmmAmazon)
    dfML = pd.read_csv(pathEmmML)

    # Crear lista de (id, idioma, embedding) para Amazon y MercadoLibre
    amazon_data = [(row['id'], row['idioma'], row[2:].values) for _, row in dfAmazon.iterrows()]
    ml_data = [(row['id'], row['idioma'], row[2:].values) for _, row in dfML.iterrows()]

    # Dividir datos en sublistas para paralelización
    print("Dividiendo datos para procesamiento paralelo...")
    sublistsAmazon = nivelacion_cargas(amazon_data, N_THREADS)
    sublistsML = nivelacion_cargas(ml_data, N_THREADS)

    # Crear encabezados en los archivos de salida
    for output_file in [outputLabelsAmazon, outputLabelsML]:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'idioma', 'label'])

    # Iniciar clasificación con threads específicos de etiquetado
    threads_lb = []

    print("Iniciando clasificación de Amazon y MercadoLibre...")
    start_time = time.perf_counter()

    # Procesar sublistas en paralelo
    for i in range(N_THREADS):
        # Threads para Amazon
        threads_lb.append(Process(target=label_worker_batch, args=(sublistsAmazon[i], lock, outputLabelsAmazon)))

        # Threads para MercadoLibre
        threads_lb.append(Process(target=label_worker_batch, args=(sublistsML[i], lock, outputLabelsML)))

    # Iniciar y unir threads
    for thread in threads_lb:
        thread.start()

    for thread in threads_lb:
        thread.join()

    finish_time = time.perf_counter()
    print(f"Clasificación terminada en {finish_time - start_time:.2f} segundos.")

    combinar_csv_por_id(outputLabelsAmazon, pathReviewsAmazon, 'outputs/AmazonEtiquetado.csv' )
    combinar_csv_por_id(outputLabelsML, pathReviewsML, 'outputs/MLEtiquetado.csv' )

    eliminar_filas_con_unknown('outputs/AmazonEtiquetado.csv', 'label')
    eliminar_filas_con_unknown('outputs/MLEtiquetado.csv', 'label')
    



