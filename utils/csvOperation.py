def addIdCsv(archivo_original, name):
    """
    Modifica un archivo CSV eliminando filas vacías y agregando una columna de ID incremental como la primera columna.
    
    :param archivo_original: Nombre del archivo CSV a modificar.
    :param name: Nombre de la columna de texto original.
    """
    import pandas as pd

    # Leer el archivo original
    df = pd.read_csv(archivo_original, names=[name])
    
    # Eliminar filas vacías y reiniciar índices
    df = df.dropna().reset_index(drop=True)
    
    # Agregar la columna de ID como la primera columna
    df.insert(0, 'id', df.index + 1)  # Insertar la columna 'id' en la posición 0
    
    # Sobrescribir el archivo original
    df.to_csv(archivo_original, index=False)
    # print(f"El archivo {archivo_original} ha sido modificado eliminando filas vacías y agregando una columna de ID.")
