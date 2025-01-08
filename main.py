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

    pathReviewsAmazon = 'scrapping/outputScrapping/reviewsAmazon.csv'
    pathReviewsML = 'scrapping/outputScrapping/reviewsML.csv'

    N_THREADS = 12
    lock = Lock()

    archivo_csv = "scrapping/outputScrapping/reviewsML.csv"
    df = pd.read_csv(archivo_csv)

    # Eliminar la columna 'id' (o el nombre de la columna específica)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Guardar el archivo CSV sin la columna 'id'
    df.to_csv("scrapping/outputScrapping/reviewsML.csv", index=False)

    def run_scraping():
        threadsScrapping = []
        
        product_name = input("\nProducto: ")
        linksML = getLinksML(product_name)
        linkscleanML = limpiar(linksML)
        linksAmazon = getLinksAmazon(product_name)

        sublistsML = nivelacion_cargas(linkscleanML, N_THREADS)
        sublistsAmazon = nivelacion_cargas(linksAmazon, N_THREADS)

        for i in range(N_THREADS):
            threadsScrapping.append(Process(target=AmazonReviews, args=(sublistsAmazon[i], lock)))
            threadsScrapping.append(Process(target=scrape_url, args=(sublistsML[i], lock)))

        print("Starting scraping...")
        start_time = time.perf_counter()

        for thread in threadsScrapping:
            thread.start()

        for thread in threadsScrapping:
            thread.join()

        finish_time = time.perf_counter()
        print(f"Scraping finished in {finish_time - start_time} seconds")

    def run_preprocessing():
        threadsProcess = []
        addIdCsv(pathReviewsAmazon, 'comentarios')
        addIdCsv(pathReviewsML, 'comentarios')

        dfAmazon = pd.read_csv(pathReviewsAmazon)
        dfML = pd.read_csv(pathReviewsML)

        amazon_reviews = list(zip(dfAmazon['id'], dfAmazon['comentarios']))
        ml_reviews = list(zip(dfML['id'], dfML['comentarios']))

        subReviewsAmazon = nivelacion_cargas(amazon_reviews, N_THREADS)
        subReviewsML = nivelacion_cargas(ml_reviews, N_THREADS)

        nltk.download('stopwords')

        for i in range(N_THREADS):
            #threadsProcess.append(Process(target=preprocessText, args=('ML', subReviewsML[i], lock)))
            threadsProcess.append(Process(target=preprocessText, args=('Amazon', subReviewsAmazon[i], lock)))

        print("Starting preprocessing...")
        start_time = time.perf_counter()

        for thread in threadsProcess:
            thread.start()

        for thread in threadsProcess:
            thread.join()

        finish_time = time.perf_counter()
        print(f"Text Processing finished in {finish_time - start_time} seconds")

    def run_labeling():
        pathEmmAmazon = 'nlp/preprocess_Amazon_Text.csv'
        pathEmmML = 'nlp/preprocess_ML_Text.csv'
        outputLabelsAmazon = 'nlp/sentimentAnalysis/labels_Amazon.csv'
        outputLabelsML = 'nlp/sentimentAnalysis/labels_ML.csv'

        print("Reading Amazon and MercadoLibre data...")
        dfAmazon = pd.read_csv(pathEmmAmazon)
        dfML = pd.read_csv(pathEmmML)

        amazon_data = [(row['id'], row['idioma'], row[2:].values) for _, row in dfAmazon.iterrows()]
        ml_data = [(row['id'], row['idioma'], row[2:].values) for _, row in dfML.iterrows()]

        print("Dividing data for parallel processing...")
        sublistsAmazon = nivelacion_cargas(amazon_data, N_THREADS)
        sublistsML = nivelacion_cargas(ml_data, N_THREADS)

        for output_file in [outputLabelsAmazon, outputLabelsML]:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'idioma', 'label'])

        threads_lb = []

        print("Starting classification...")
        start_time = time.perf_counter()

        for i in range(N_THREADS):
            threads_lb.append(Process(target=label_worker_batch, args=(sublistsAmazon[i], lock, outputLabelsAmazon)))
            threads_lb.append(Process(target=label_worker_batch, args=(sublistsML[i], lock, outputLabelsML)))

        for thread in threads_lb:
            thread.start()

        for thread in threads_lb:
            thread.join()

        finish_time = time.perf_counter()
        print(f"Classification finished in {finish_time - start_time:.2f} seconds")

        # Final file operations
        combinar_csv_por_id(outputLabelsAmazon, pathReviewsAmazon, 'outputs/AmazonEtiquetado.csv')
        combinar_csv_por_id(outputLabelsML, pathReviewsML, 'outputs/MLEtiquetado.csv')

        eliminar_filas_con_unknown('outputs/AmazonEtiquetado.csv', 'label')
        eliminar_filas_con_unknown('outputs/MLEtiquetado.csv', 'label')

    # Run the processes sequentially
    #run_scraping()
    #print("\nScraping complete. Starting preprocessing phase...")
    run_preprocessing()
    print("\nPreprocessing complete. Starting labeling phase...")
    run_labeling()
    print("\nAll processes completed successfully!")