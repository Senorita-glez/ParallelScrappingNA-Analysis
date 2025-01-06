if __name__ == '__main__':
    from scrapping.amazon import *
    from scrapping.mercadoLibre import *
    from utils.paralel import nivelacion_cargas
    from multiprocessing import Lock, Process
    from nlp.textPreprocess import preprocessText
    import time 
    import pandas as pd

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


    dfAmazon = pd.read_csv('scrapping/outputScrapping/reviewsAmazon.csv', names=['comentarios'])
    dfML = pd.read_csv('scrapping/outputScrapping/reviewsML.csv', names=['comentarios'])
    
    amazon_reviews = dfAmazon['comentarios'].tolist()
    ml_reviews = dfML['comentarios'].tolist()
    subReviewsAmazon = nivelacion_cargas(amazon_reviews, N_THREADS)
    subReviewsML = nivelacion_cargas(ml_reviews, N_THREADS)

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
    print(f"Preprocess finished in {finish_time - start_time} seconds")




