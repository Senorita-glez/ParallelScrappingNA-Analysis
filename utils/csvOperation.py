import pandas as pd

def addIdCsv(archivo_original, name):
    """
    Modifica un archivo CSV eliminando filas vacías y agregando una columna de ID incremental como la primera columna.
    
    :param archivo_original: Nombre del archivo CSV a modificar.
    :param name: Nombre de la columna de texto original.
    """

    # Leer el archivo original respetando las comillas para valores con comas
    df = pd.read_csv(archivo_original, names=[name], quotechar='"', engine='python')
    
    # Eliminar filas vacías y reiniciar índices
    df = df.dropna().reset_index(drop=True)
    
    # Agregar la columna de ID como la primera columna
    df.insert(0, 'id', df.index + 1)  # Insertar la columna 'id' en la posición 0
    
    # Sobrescribir el archivo original respetando las comillas
    df.to_csv(archivo_original, index=False, quotechar='"', quoting=1)  # quoting=1 asegura que las comillas se respeten


def combinar_csv_por_id(csv1_path, csv2_path, output_path):
    """
    Combina dos archivos CSV por la columna 'id' y guarda el resultado en el segundo archivo.

    Args:
        csv1_path (str): Ruta del primer archivo CSV.
        csv2_path (str): Ruta del segundo archivo CSV.
        output_path (str): Ruta donde se guardará el archivo combinado.

    Returns:
        None
    """
    # Leer los archivos CSV
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Combinar los DataFrames por la columna 'id'
    combinado = pd.merge(df1, df2, on='id', how='inner')

    # Guardar el archivo combinado en el segundo CSV
    combinado.to_csv(output_path, index=False)

    print(f"Archivos combinados y guardados en {output_path}.")


def eliminar_filas_con_unknown(path, columna):
    """
    Elimina las filas de un archivo CSV donde la columna especificada tenga el valor 'unknown'.
    Sobrescribe el archivo original con el resultado.

    Args:
        path (str): Ruta del archivo CSV.
        columna (str): Nombre de la columna a verificar.

    Returns:
        None
    """
    # Leer el archivo CSV
    df = pd.read_csv(path)

    # Filtrar filas donde la columna no sea 'unknown'
    df_filtrado = df[df[columna] != 'unknown']

    # Sobrescribir el archivo original con las filas filtradas
    df_filtrado.to_csv(path, index=False)

    print(f"Filas con 'unknown' en la columna '{columna}' eliminadas. Archivo sobrescrito en {path}.")


def ordenar_csv_por_id_en_mismo_archivo(path, columna_id='id'):
    """
    Ordena un archivo CSV por la columna 'id' de menor a mayor y sobrescribe el mismo archivo.

    Args:
        path (str): Ruta del archivo CSV.
        columna_id (str): Nombre de la columna por la cual se ordenará el archivo (por defecto es 'id').

    Returns:
        None
    """
    # Leer el archivo CSV
    df = pd.read_csv(path)

    # Verificar si la columna existe
    if columna_id not in df.columns:
        raise ValueError(f"La columna '{columna_id}' no existe en el archivo.")

    # Ordenar el DataFrame por la columna 'id'
    df_ordenado = df.sort_values(by=columna_id)

    # Sobrescribir el archivo original
    df_ordenado.to_csv(path, index=False)

    print(f"Archivo ordenado por '{columna_id}' y sobrescrito en {path}.")

