# Cargar el dataset
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/ignaciomsarmiento/RecomSystemsLectures/main/L07_sentimientos/data/Amazon.csv', index_col="Unnamed: 0")

# Ver las primeras filas del dataset
print(data.head())

# Verificar las columnas y tipos de datos
print("\nColumnas y tipos de datos:")
print(data.info())

# Revisar la distribución de la etiqueta de sentimiento (ajustar 'label_column' si es necesario)
print("\nDistribución de etiquetas de sentimiento:")
print(data['stars'].value_counts())
