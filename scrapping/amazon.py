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
        'accept-language': 'en-GB,en;q=0.9', # Mejorar los headers
    }

    # Lista para almacenar los enlaces y precios de los productos
    products = []
    num_products = 10  # Número de productos a recolectar
    current_url = url  # URL inicial para comenzar la búsqueda

    while len(products) < num_products:
        try:
            response = requests.get(current_url, params=params, headers=custom_headers)

            # Comprobamos que la solicitud fue exitosa
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Imprimir el título de la página
                page_title = soup.title.string.strip() if soup.title else "Sin título"
                print(f"Visitando página: {page_title}")

                # Encontrar enlaces de productos con la clase especificada
                product_links = soup.find_all(
                    "a", 
                    class_="a-link-normal s-line-clamp-4 s-link-style a-text-normal"
                )

                # Encontrar precios de los productos
                product_prices = soup.find_all("span", class_="a-price-whole")

                for product, price in zip(product_links, product_prices):
                    if len(products) >= num_products:
                        break

                    link = "https://www.amazon.com.mx" + product['href']
                    price_text = price.get_text(strip=True).replace(".", "")  # Limpiar el precio
                    products.append({"link": link, "price": price_text})

                # Buscar enlace de la siguiente página
                next_page = soup.find("a", class_="s-pagination-item s-pagination-next s-pagination-button s-pagination-button-accessibility s-pagination-separator")

                if next_page and 'href' in next_page.attrs:
                    current_url = "https://www.amazon.com.mx" + next_page['href']
                else:
                    print("No hay más páginas disponibles o ya se han recolectado todos los productos.")
                    break

            else:
                print(f"Error al acceder a la página. Código de estado: {response.status_code}")
                return None

        except Exception as e:
            print(f"Ocurrió un error al intentar acceder a la página: {e}")
            return None

    # Guardar los datos en un archivo CSV
    with open("productos.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["link", "price"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(products)

    print("Datos guardados en productos.csv")
    return products

def getUserNPassword():
    import csv
    import random
    with open('notYourBusiness.csv', mode='r', newline='') as file:
        rows = list(csv.DictReader(file))

    rows_with_status_0 = [row for row in rows if row['status'] == '0']
    if rows_with_status_0:
        selected_row = random.choice(rows_with_status_0)
        #for row in rows:
            #if row['id'] == selected_row['id']:
                #row['status'] = '1'
                #break

        with open('notYourBusiness.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        return selected_row['id'], selected_row['u'], selected_row['p']

    return None, None, None

def LogInAmazon(driver, wait, email, password, lock):
    import time
    import random
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    try:
        if "Iniciar sesión" in driver.title:
            email_input = wait.until(EC.presence_of_element_located((By.ID, "ap_email")))
            email_input.send_keys(email)
            email_input.send_keys(Keys.RETURN)
            time.sleep(random.uniform(1, 3))

            password_input = wait.until(EC.presence_of_element_located((By.ID, "ap_password")))
            password_input.send_keys(password)
            password_input.send_keys(Keys.RETURN)
            time.sleep(random.uniform(1, 3))
    except Exception as e:
        lock.acquire()
        print(f"Error during login: {e}")
        lock.release()

def initialize_browser(lock):
    from selenium import webdriver
    import user_agent
    options = webdriver.ChromeOptions()
    user_agent_string = user_agent.generate_user_agent()
    options.add_argument(f"user-agent={user_agent_string}")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-extensions")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option('excludeSwitches',['enable-logging'])
    options.add_argument('--log-level=3')

    try:
        driver = webdriver.Chrome(options=options)
        return driver
    except Exception as e:
        lock.acquire()
        print(f"Error initializing Chrome driver: {e}")
        lock.release()
        return None

def access_reviews_with_auto_login(product_url, lock):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time
    import random
    import csv

    idU, email, password = getUserNPassword()
    driver = initialize_browser(lock)
    if driver is None:
        return

    wait = WebDriverWait(driver, 20)  # Increased wait time to 20 seconds
    all_reviews = []

    try:
        driver.get(product_url['link'])
        time.sleep(random.uniform(1, 3))
        LogInAmazon(driver, wait, email, password, lock)
        
        try:
            see_all_reviews_link = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-hook="see-all-reviews-link-foot"]'))
            )
            see_all_reviews_link.click()
            time.sleep(random.uniform(1, 3))
        except Exception as e:
            lock.acquire()
            print(f"Error clicking 'see all reviews' link for {product_url} (Page title: {driver.title})")
            lock.release()
            return

        LogInAmazon(driver, wait, email, password, lock)

        while True:
            try:
                LogInAmazon(driver, wait, email, password, lock)
                review_elements = wait.until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'span[data-hook="review-body"]'))
                )
                for review in review_elements:
                    all_reviews.append(review.text)
                
                time.sleep(random.uniform(1, 3))
                
                next_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'li.a-last a')))
                next_button.click()
                time.sleep(random.uniform(1, 3))
            except Exception as e:
                lock.acquire()
                print(f"Error navigating to the next page for {product_url} (Page title: {driver.title}): {e}")
                lock.release()
                break

        lock.acquire()
        try:
            with open("reviews_dataAmazon.csv", mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                for review in all_reviews:
                    writer.writerow([review])
            print(f"Saved reviews for {product_url}")
        except Exception as e:

            print(f"Error saving reviews for {product_url} (Page title: {driver.title}): {e}")
        finally:
            lock.release()

    except Exception as e:
        lock.acquire()
        print(f"An error occurred for {product_url} (Page title: {driver.title}): {e}")
        lock.release()

    finally:
        driver.save_screenshot("error_screenshot.png")  # Save a screenshot for debugging
        print(driver.page_source)  # Print page source for debugging
        driver.quit()


def AmazonReviews(sublistAmazonReviews, lock):
    for product_url in sublistAmazonReviews:
        access_reviews_with_auto_login(product_url, lock)
