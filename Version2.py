'''
NIVELACION CARGAS
'''

def nivelacion_cargas(D, n_p):
    s = len(D)%n_p
    n_D = D[:s]
    t = int((len(D)-s)/n_p)
    out=[]
    temp=[]
    for i in D[s:]:
        temp.append(i)
        if len(temp)==t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

'''
MERCADO LIBRE LINKS
'''
def getLinksML(product_name):
    from bs4 import BeautifulSoup
    import requests
    import pandas as pd
    base_url = 'https://listado.mercadolibre.com.mx/'
    cleaned_name = product_name.replace(" ", "-").lower()
    urls = [base_url + cleaned_name]

    page_number = 50
    for i in range(0, 10000, 50):
        urls.append(f"{base_url}{cleaned_name}_Desde_{page_number + 1}_NoIndex_True")
        page_number += 50

    data = []
    c = 1
        
    for i, url in enumerate(urls, start=1):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find_all('li', class_='ui-search-layout__item')
        
        if not content:
            break

        for post in content:
            post_link = post.find("a")["href"]
            post_data = {
                # "title": title,
            #     "price": price,
                "post link": post_link,        
            }
            data.append(post_data)
            c += 1
    datalinks = pd.DataFrame(data)
    return datalinks

def limpiar(dataframe):
    import re
    import pandas as pd

    def extract_code(link):
        # Extraer el código entre '/p/' y '#polycard'
        match = re.search(r'/p/([^#]*)#polycard', link)
        if match:
            return match.group(1)
        match_mlm = re.search(r'mx/MLM-(\d+)-', link)
        if match_mlm:
            return f"MLM{match_mlm.group(1)}"
        return None

    def create_new_link(code):
        if code:
            return f"https://www.mercadolibre.com.mx/noindex/catalog/reviews/{code}?noIndex=true&access=view_all&modal=true&controlled=true"
        return None

    dataframe['CodigoExtraido'] = dataframe['post link'].apply(extract_code)
    dataframe['NuevoLink'] = dataframe['CodigoExtraido'].apply(create_new_link)
    df_final = dataframe[['NuevoLink']].dropna()

    # Save the final dataframe to a CSV file
    df_final.to_csv("ML_data666.csv", index=False)
    print('se logro')

    # Return a list of links
    link_list = df_final['NuevoLink'].tolist()
    return link_list

'''
MERCADO LIBRE ONE LINK
'''
def scrape_url(reviews_urls, lock):
    import requests
    from bs4 import BeautifulSoup
    import csv

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    reviews_texts = []

    for reviews_url in reviews_urls:
        try:
            response = requests.get(reviews_url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                reviews = soup.find_all("p", {"class": "ui-review-capability-comments__comment__content ui-review-capability-comments__comment__content"})  # Verifica la clase correcta
                reviews_texts.extend([review.get_text().strip() for review in reviews])
            else:
                print(f"No se pudo acceder a la página de reseñas para la URL: {reviews_url}")
        except Exception as e:
            print(f"Error al procesar la URL {reviews_url}: {e}")

    # Write reviews to a CSV file with thread safety
    lock.acquire()
    try:
        with open("reviews_dataML.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for review in reviews_texts:
                writer.writerow([review])
    finally:
        lock.release()


'''
AMAZON LINKS
'''
def getLinksAmazon(product_name):
    import requests
    from bs4 import BeautifulSoup
    from user_agent import generate_user_agent
    import csv
    url = "https://www.amazon.com.mx/s"  # URL de búsqueda en Amazon México

    # Parámetros para la búsqueda
    params = {
        'k': product_name  # Término de búsqueda, en este caso "freidora"
    }

    # Encabezados personalizados
    custom_headers = {
        'user-agent': generate_user_agent(),  # Generar un User-Agent aleatorio
        'accept-language': 'en-GB,en;q=0.9', # mejorar los headers 
    }

    # Lista para almacenar los enlaces de los productos
    product_links_list = []
    num_products = 500  # Número de productos a recolectar
    current_url = url  # URL inicial para comenzar la búsqueda

    while len(product_links_list) < num_products:
        response = requests.get(current_url, params=params, headers=custom_headers)
        
        # Comprobamos que la solicitud fue exitosa
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            product_links = soup.find_all("a", class_="a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal")

            for product in product_links:
                if len(product_links_list) >= num_products:
                    break

                link = "https://www.amazon.com.mx" + product['href']
                product_links_list.append(link)
            
            next_page = soup.find("a", class_="s-pagination-item s-pagination-next s-pagination-button s-pagination-separator")
            
            if next_page and 'href' in next_page.attrs:
                current_url = "https://www.amazon.com.mx" + next_page['href']
            else:
                print("yap quedo o no hay massss ")
                with open(f"{product_name}_product_links.csv", mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Product Link"])  # Write header
                    for link in product_links_list:
                        writer.writerow([link])
                return product_links_list
        else:
            print("No se pudo acceder a la página. Status code:", response.status_code)
            return None
        



if __name__ == '__main__':
    import multiprocess, time, multiprocessing

    threads = []
    N_THREADS = 6
    
    product_name = input("\nProducto: ")
    linksML = getLinksML(product_name)
    linkscleanML = limpiar(linksML)
    linksAmazon = getLinksAmazon(product_name)

    sublistsML = nivelacion_cargas(linkscleanML, N_THREADS)
    sublistsAmazon = nivelacion_cargas(linksAmazon, N_THREADS)

    lock = multiprocess.Lock()
    for i in range(N_THREADS):
        # Initialize each process with a sublist of URLs
        threads.append(multiprocess.Process(target=scrape_url, args=(sublistsML[i], lock)))

    start_time = time.perf_counter()
    
    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
                
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")


