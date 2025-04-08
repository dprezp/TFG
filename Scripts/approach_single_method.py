import os
import gc
import argparse
import time
import copy
import numpy as np
import pandas as pd
from sklearn import tree
from utility import get_data
from funciones_dt_prune import (
    get_metrics, best_improvement_prune, first_improvement_prune, get_state_of_art_algorithm
)
from funciones_dt_relabeling import (
    first_improvement_relabeling, best_improvement_relabeling
)
from funciones_df import table_align, write_metrics

# Configurar entorno para evitar conflicto de threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Configurar argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--method", type=str, required=True,
                    choices=["state_of_art", "first_prune", "best_prune",
                             "first_relabel", "best_relabel", "all"],
                    help="Method to execute")
args = parser.parse_args()

# Definir datasets y atributos protegidos
DATASETS = {
    "adult": ["sex", "race"],
    "german": ["sex", "age"],
    "compas": ["race", "sex"],
    "bank": ["age"]
}

METHODS = {
    "state_of_art": get_state_of_art_algorithm,
    "first_prune": first_improvement_prune,
    "best_prune": best_improvement_prune,
    "first_relabel": first_improvement_relabeling,
    "best_relabel": best_improvement_relabeling
}


def execute_method(dataset_used, attr, method):
    # Cargar datos
    dataset_orig, privileged_groups, unprivileged_groups, _ = get_data(dataset_used, attr)
    length_data = dataset_orig.features.shape[0]

    # Split datos
    np.random.seed(1234)
    train, test = dataset_orig.split([0.7], shuffle=True)
    test, valid = test.split([0.5], shuffle=True)

    # Inicializar DataFrames
    metrics_df = pd.DataFrame(columns=[
        "Tree Hash", "Operator", "Dataset Used", "Lenght Dataset", "Attribute",
        "Train Accuracy", "Train AOD", "Test Accuracy", "Test AOD",
        "Validation Accuracy", "Validation AOD", "Num Nodes", "Prune Count", "Elapsed Time (s)"
    ])

    grafic_df = pd.DataFrame(columns=["DataSet", "Atribute", "Operator", "Fairness", "Time"])
    max_time = [-1]
    data_tuple = (grafic_df, max_time)

    # Entrenar árbol base
    clf = tree.DecisionTreeClassifier(random_state=1)
    clf.fit(train.features, train.labels)

    # Configurar histórico inicial
    hist = [(0, 0, 0)]  # Histórico dummy

    start_time = time.time()

    # Ejecutar método seleccionado
    if method == "all":
        for m in METHODS.values():
            current_clf = copy.deepcopy(clf)
            current_clf, _ = m(
                current_clf, valid, valid,
                unprivileged_groups, privileged_groups,
                hist, data_tuple, dataset_used, attr
            )
    else:
        clf, _ = METHODS[method](
            clf, valid, valid,
            unprivileged_groups, privileged_groups,
            hist, data_tuple, dataset_used, attr
        )

    elapsed_time = time.time() - start_time

    # Escribir métricas
    metrics_df = write_metrics(
        clf, train, train, unprivileged_groups, privileged_groups,
        test, test, valid, valid, method, hist, len(clf.tree_.children_left),
        dataset_used, elapsed_time, attr, metrics_df, length_data
    )

    # Guardar resultados
    os.makedirs("Results", exist_ok=True)
    metrics_df.to_excel(f"Results/metrics_{method}_{dataset_used}_{attr}.xlsx")


# Lógica principal
if __name__ == "__main__":
    for dataset, attributes in DATASETS.items():
        for attr in attributes:
            print(f"Procesando: {dataset} - {attr}")
            if args.method == "all":
                for method_name in METHODS.keys():
                    execute_method(dataset, attr, method_name)
            else:
                execute_method(dataset, attr, args.method)
    print("Proceso completado!")