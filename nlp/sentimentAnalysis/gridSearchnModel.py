import itertools
import multiprocess
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from joblib import dump

n_cores = 10

def nivelacion_de_cargas(n_cores, lista_inicial):
    lista_final = []
    longitud_li = len(lista_inicial)
    carga = longitud_li // n_cores
    salidas = longitud_li % n_cores
    contador = 0

    for i in range(n_cores):
        if i < salidas:
            carga2 = contador + carga + 1
        else:
            carga2 = contador + carga
        lista_final.append(lista_inicial[contador:carga2])
        contador = carga2
    return lista_final

# Definir parámetros para grid search
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 0.001, 0.01]
}

param_grid_rn = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'sgd']
}

keys_svm, values_svm = zip(*param_grid_svm.items())
combinations_svm = [dict(zip(keys_svm, v)) for v in itertools.product(*values_svm)]

keys_rn, values_rn = zip(*param_grid_rn.items())
combinations_rn = [dict(zip(keys_rn, v)) for v in itertools.product(*values_rn)]

def evaluate_svm(hyperparameter_set, mejor_result, lock):
    df = pd.read_csv('../embeddings_con_etiqueta_en.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

    for s in hyperparameter_set:
        clf = SVC()
        clf.set_params(C=s['C'], kernel=s['kernel'], gamma=s['gamma'])
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        proce_accuracy = accuracy_score(y_test, y_pred)

        lock.acquire()
        if proce_accuracy > mejor_result['accuracy']:
            mejor_result['accuracy'] = proce_accuracy
            mejor_result['params'] = s
        lock.release()

def evaluate_rn(hyperparameter_set, mejor_result, lock):
    df = pd.read_csv('../embeddings_con_etiqueta_en.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

    for s in hyperparameter_set:
        clf = MLPClassifier()
        clf.set_params(hidden_layer_sizes=s['hidden_layer_sizes'], activation=s['activation'], solver=s['solver'])
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        proce_accuracy = accuracy_score(y_test, y_pred)

        lock.acquire()
        if proce_accuracy > mejor_result['accuracy']:
            mejor_result['accuracy'] = proce_accuracy
            mejor_result['params'] = s
        lock.release()

def evaluate_nb(mejor_result, lock):
    df = pd.read_csv('../embeddings_con_etiqueta_en.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    proce_accuracy = accuracy_score(y_test, y_pred)

    lock.acquire()
    if proce_accuracy > mejor_result['accuracy']:
        mejor_result['accuracy'] = proce_accuracy
    lock.release()

if __name__ == '__main__':
    threads = []
    lock = multiprocess.Lock()

    with multiprocess.Manager() as manager:
        mejor_result_svm = manager.dict({'accuracy': 0, 'params': None})
        mejor_result_rn = manager.dict({'accuracy': 0, 'params': None})
        mejor_result_nb = manager.dict({'accuracy': 0})

        start_time = time.perf_counter()

        splits_svm = nivelacion_de_cargas(n_cores, combinations_svm)
        splits_rn = nivelacion_de_cargas(n_cores, combinations_rn)

        for i in range(n_cores):
            threads.append(multiprocess.Process(target=evaluate_svm, args=(splits_svm[i], mejor_result_svm, lock)))
            threads.append(multiprocess.Process(target=evaluate_rn, args=(splits_rn[i], mejor_result_rn, lock)))

        threads.append(multiprocess.Process(target=evaluate_nb, args=(mejor_result_nb, lock)))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        finish_time = time.perf_counter()

        print(f"\nMejor SVM: {mejor_result_svm}")
        print(f"Mejor RN: {mejor_result_rn}")
        print(f"Mejor NB: {mejor_result_nb['accuracy']}")
        print(f"\nTiempo: {finish_time - start_time:.2f} segundos")

        # Entrenar y guardar modelos
        df = pd.read_csv('../embeddings_con_etiqueta_en.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

        # SVM

        svm_model = SVC(**mejor_result_svm['params'])
        svm_model.fit(X_train, y_train)
        dump(svm_model, 'svm_model.joblib')

        # RN
        rn_model = MLPClassifier(**mejor_result_rn['params'])
        rn_model.fit(X_train, y_train)
        dump(rn_model, 'rn_model.joblib')

        # NB
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        dump(nb_model, 'nb_model.joblib')

        print("Modelos guardados exitosamente.")