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
    import os
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_output = os.path.join(ruta_actual, "outputScrapping", "linksML.csv")

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
    df_final.to_csv(ruta_output, index=False)
    print('se logro')

    # Return a list of links
    link_list = df_final['NuevoLink'].tolist()
    return link_list

'''
MERCADO LIBRE ONE LINK
'''
def scrape_url(sublistMLReviews, lock):
    import requests
    from bs4 import BeautifulSoup
    import csv
    import os
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_output = os.path.join(ruta_actual, "outputScrapping", "reviewsML.csv")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def get_reviews(product_url):
        reviews_texts = []
        try:
            response = requests.get(product_url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                reviews = soup.find_all("p", {"class": "ui-review-capability-comments__comment__content ui-review-capability-comments__comment__content"})
                reviews_texts = [review.get_text().strip() for review in reviews]
            else:
                print(f"No se pudo acceder a la página de reseñas para la URL: {product_url}")
        except Exception as e:
            print(f"Error al procesar la URL {product_url}: {e}")
        return reviews_texts

    for product_url in sublistMLReviews:
        reviews_texts = get_reviews(product_url)
        lock.acquire()
        try:
            with open(ruta_output, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                for review in reviews_texts:
                    writer.writerow([review])
        finally:
            lock.release()