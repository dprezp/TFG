import os
import gc
import copy
import argparse
import time
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

# =============================================================================
# ----------------------- CÓDIGO COPIADO DE approach_dt.py --------------------
# =============================================================================

# Definir datasets y atributos protegidos
datasets = ["adult", "german", "compas", "bank"]
protected_attributes = [["sex", "race"], ["sex", "age"], ["race", "sex"], ["age"]]

# Inicializar DataFrames (igual que en approach_dt.py)
metrics_prune_df = pd.DataFrame(columns=[
    "Tree Hash", "Operator", "Dataset Used", "Lenght Dataset", "Attribute",
    "Train Accuracy", "Train AOD", "Test Accuracy", "Test AOD",
    "Validation Accuracy", "Validation AOD", "Num Nodes", "Prune Count", "Elapsed Time (s)"
])
metrics_relabeling_df = pd.DataFrame(columns=[
    "Tree Hash", "Operator", "Dataset Used", "Lenght Dataset", "Attribute",
    "Train Accuracy", "Train AOD", "Test Accuracy", "Test AOD",
    "Validation Accuracy", "Validation AOD", "Num Nodes", "Prune Count", "Elapsed Time (s)"
])

grafic_df = pd.DataFrame(columns=["DataSet", "Atribute", "Operator", "Fairness", "Time"])
max_time = [-1]
data_tuple = (grafic_df, max_time)
first_fairness = {}


# =============================================================================
# ----------------------- FUNCIÓN MODIFICADA CON SELECCIÓN DE MÉTODO ----------
# =============================================================================

def process_dataset(dataset_index, dataset_used, attr):
    global metrics_prune_df, metrics_relabeling_df, data_tuple

    # Código original de approach_dt.py
    dataset_orig, privileged_groups, unprivileged_groups, _ = get_data(dataset_used, attr)
    length_data = dataset_orig.features.shape[0]

    np.random.seed(1234)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_test, dataset_orig_valid = dataset_orig_test.split([0.5], shuffle=True)

    # Copias exactas como en approach_dt.py
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)

    clf = tree.DecisionTreeClassifier(random_state=1)
    clf.fit(dataset_orig_train.features, dataset_orig_train.labels)

    # Histórico inicial REAL (idéntico a approach_dt.py)
    valid_acc, valid_aod = get_metrics(clf, dataset_orig_valid, dataset_orig_valid_pred,
                                       unprivileged_groups, privileged_groups)
    valid_fair = valid_aod
    prune_count = 0
    hist = [(valid_acc, valid_fair, prune_count)]

    # =========================================================================
    # ----------------------- LÓGICA DE SELECCIÓN DE MÉTODO -------------------
    # =========================================================================

    operators = {
        "state_of_art": (get_state_of_art_algorithm, "prune"),
        "first_prune": (first_improvement_prune, "prune"),
        "best_prune": (best_improvement_prune, "prune"),
        "first_relabel": (first_improvement_relabeling, "relabel"),
        "best_relabel": (best_improvement_relabeling, "relabel")
    }

    if args.method == "all":
        methods_to_run = operators.keys()
    else:
        methods_to_run = [args.method]

    for method in methods_to_run:
        current_clf = copy.deepcopy(clf)
        current_hist = copy.deepcopy(hist)

        start_time = time.time()

        # Ejecutar método seleccionado
        if method == "state_of_art":
            current_clf, data_tuple = operators[method][0](
                current_clf, 2500, len(current_clf.tree_.children_left), prune_count,
                dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,
                privileged_groups, current_hist, data_tuple, dataset_used, attr
            )
        else:
            current_clf, data_tuple = operators[method][0](
                current_clf, dataset_orig_valid, dataset_orig_valid_pred,
                unprivileged_groups, privileged_groups, current_hist,
                data_tuple, dataset_used, attr, 0, None
            )

        elapsed_time = time.time() - start_time

        # Escribir métricas (idéntico a approach_dt.py)
        if operators[method][1] == "prune":
            metrics_prune_df = write_metrics(
                current_clf, dataset_orig_train, dataset_orig_train_pred,
                unprivileged_groups, privileged_groups, dataset_orig_test,
                dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred,
                method, current_hist, len(current_clf.tree_.children_left),
                dataset_used, elapsed_time, attr, metrics_prune_df, length_data
            )
        else:
            metrics_relabeling_df = write_metrics(
                current_clf, dataset_orig_train, dataset_orig_train_pred,
                unprivileged_groups, privileged_groups, dataset_orig_test,
                dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred,
                method, current_hist, len(current_clf.tree_.children_left),
                dataset_used, elapsed_time, attr, metrics_relabeling_df, length_data
            )


# =============================================================================
# ----------------------- BUCLE PRINCIPAL (MODIFICADO) ------------------------
# =============================================================================

if __name__ == "__main__":
    for dataset_index, dataset_used in enumerate(datasets):
        for attr in protected_attributes[dataset_index]:
            print(f"Procesando: {dataset_used} - {attr}")
            process_dataset(dataset_index, dataset_used, attr)

    # Guardar resultados (igual que en approach_dt.py)
    metrics_prune_df.to_excel(os.path.join("Results", "metrics_Prune_results.xlsx"))
    metrics_relabeling_df.to_excel(os.path.join("Results", "metrics_Relabeling_results.xlsx"))

    prueba_df, _ = data_tuple
    prueba_df.to_excel(os.path.join("Results", "prueba.xlsx"))

    grafic_df = table_align(data_tuple, first_fairness)
    grafic_df.to_excel(os.path.join("Results", "grafic_metrics.xlsx"))

    print("Proceso completado!")