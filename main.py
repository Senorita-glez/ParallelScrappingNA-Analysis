if __name__ == '__main__':
    from scrapping.amazon import *
    from scrapping.mercadoLibre import *
    from utils.paralel import nivelacion_cargas
    from multiprocessing import Lock, Process
    import time 
    
    threads = []
    N_THREADS = 4

    product_name = input("\nProducto: ")
    linksML = getLinksML(product_name)
    linkscleanML = limpiar(linksML)
    #linksAmazon = getLinksAmazon(product_name)

    sublistsML = nivelacion_cargas(linkscleanML, N_THREADS)
    #sublistsAmazon = nivelacion_cargas(linksAmazon, N_THREADS)

    lock = Lock()
    for i in range(N_THREADS):
        # Initialize each process for Mercado Libre and Amazon reviews
        threads.append(Process(target=scrape_url, args=(sublistsML[i], lock)))
        #threads.append(Process(target=AmazonReviews, args=(sublistsAmazon[i], lock)))

    start_time = time.perf_counter()

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")