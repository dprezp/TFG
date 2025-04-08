import os
import gc
import argparse
import time
import numpy as np
import pandas as pd

from utility import get_data
from sklearn import tree
from funciones_dt_prune import (
    get_metrics, best_improvement_prune, first_improvement_prune, get_state_of_art_algorithm
)
from funciones_dt_relabeling import (
    first_improvement_relabeling, best_improvement_relabeling
)
from funciones_df import table_align, write_metrics

def train_base_tree(dataset_used, attr):
    dataset_orig, priv, unpriv, _ = get_data(dataset_used, attr)
    length = dataset_orig.features.shape[0]
    np.random.seed(1234)
    train, test = dataset_orig.split([0.7], shuffle=True)
    test, valid = test.split([0.5], shuffle=True)

    train_pred = train.copy(deepcopy=True)
    test_pred = test.copy(deepcopy=True)
    valid_pred = valid.copy(deepcopy=True)

    clf = tree.DecisionTreeClassifier(random_state=1)
    clf.fit(train.features, train.labels)

    return clf, train, test, valid, train_pred, test_pred, valid_pred, priv, unpriv, length

def run_state_of_art():
    metrics_df = pd.DataFrame()
    datasets = ["adult", "german", "compas", "bank"]
    protected_attributes = [["sex", "race"], ["sex", "age"], ["race", "sex"], ["age"]]
    grafic_df = pd.DataFrame(columns=["DataSet", "Atribute", "Operator", "Fairness", "Time"])
    max_time = [-1]
    data_tuple = (grafic_df, max_time)

    for dataset_index, dataset_used in enumerate(datasets):
        for attr in protected_attributes[dataset_index]:
            clf, train, test, valid, train_pred, test_pred, valid_pred, unpriv, priv, length = train_base_tree(dataset_used, attr)
            operator = "State_of_Art"
            hist = []
            prune_count = 0
            n_nodes = len(clf.tree_.children_left)
            start = time.time()
            clf, data_tuple = get_state_of_art_algorithm(clf, 2500, n_nodes, prune_count, valid, valid_pred, unpriv, priv, hist, data_tuple, dataset_used, attr)
            elapsed = time.time() - start
            n_nodes = len(clf.tree_.children_left)
            metrics_df = write_metrics(clf, train, train_pred, unpriv, priv, test, test_pred, valid, valid_pred, operator, hist, n_nodes, dataset_used, elapsed, attr, metrics_df, length)
            del clf, hist
            gc.collect()

    metrics_df.to_excel(os.path.join("Results", "metrics_State_of_Art.xlsx"))

def run_first_improvement_prune():
    metrics_df = pd.DataFrame()
    datasets = ["adult", "german", "compas", "bank"]
    protected_attributes = [["sex", "race"], ["sex", "age"], ["race", "sex"], ["age"]]
    grafic_df = pd.DataFrame(columns=["DataSet", "Atribute", "Operator", "Fairness", "Time"])
    max_time = [-1]
    data_tuple = (grafic_df, max_time)

    for dataset_index, dataset_used in enumerate(datasets):
        for attr in protected_attributes[dataset_index]:
            clf, train, test, valid, train_pred, test_pred, valid_pred, unpriv, priv, length = train_base_tree(dataset_used, attr)
            operator = "First_improvement"
            hist = []
            start = time.time()
            clf, data_tuple = first_improvement_prune(clf, valid, valid_pred, unpriv, priv, hist, data_tuple, dataset_used, attr, 0, None)
            elapsed = time.time() - start
            n_nodes = len(clf.tree_.children_left)
            metrics_df = write_metrics(clf, train, train_pred, unpriv, priv, test, test_pred, valid, valid_pred, operator, hist, n_nodes, dataset_used, elapsed, attr, metrics_df, length)
            del clf, hist
            gc.collect()

    metrics_df.to_excel(os.path.join("Results", "metrics_First_Improvement_Prune.xlsx"))

def run_best_improvement_prune():
    metrics_df = pd.DataFrame()
    datasets = ["adult", "german", "compas", "bank"]
    protected_attributes = [["sex", "race"], ["sex", "age"], ["race", "sex"], ["age"]]
    grafic_df = pd.DataFrame(columns=["DataSet", "Atribute", "Operator", "Fairness", "Time"])
    max_time = [-1]
    data_tuple = (grafic_df, max_time)

    for dataset_index, dataset_used in enumerate(datasets):
        for attr in protected_attributes[dataset_index]:
            clf, train, test, valid, train_pred, test_pred, valid_pred, unpriv, priv, length = train_base_tree(dataset_used, attr)
            operator = "Best_improvement"
            hist = []
            start = time.time()
            clf, data_tuple = best_improvement_prune(clf, valid, valid_pred, unpriv, priv, hist, data_tuple, dataset_used, attr)
            elapsed = time.time() - start
            n_nodes = len(clf.tree_.children_left)
            metrics_df = write_metrics(clf, train, train_pred, unpriv, priv, test, test_pred, valid, valid_pred, operator, hist, n_nodes, dataset_used, elapsed, attr, metrics_df, length)
            del clf, hist
            gc.collect()

    metrics_df.to_excel(os.path.join("Results", "metrics_Best_Improvement_Prune.xlsx"))

def run_first_improvement_relabeling():
    metrics_df = pd.DataFrame()
    datasets = ["adult", "german", "compas", "bank"]
    protected_attributes = [["sex", "race"], ["sex", "age"], ["race", "sex"], ["age"]]
    grafic_df = pd.DataFrame(columns=["DataSet", "Atribute", "Operator", "Fairness", "Time"])
    max_time = [-1]
    data_tuple = (grafic_df, max_time)

    for dataset_index, dataset_used in enumerate(datasets):
        for attr in protected_attributes[dataset_index]:
            clf, train, test, valid, train_pred, test_pred, valid_pred, unpriv, priv, length = train_base_tree(dataset_used, attr)
            operator = "First_improvement"
            hist = []
            start = time.time()
            clf, data_tuple = first_improvement_relabeling(clf, valid, valid_pred, unpriv, priv, hist, data_tuple, dataset_used, attr, 0, None)
            elapsed = time.time() - start
            n_nodes = len(clf.tree_.children_left)
            metrics_df = write_metrics(clf, train, train_pred, unpriv, priv, test, test_pred, valid, valid_pred, operator, hist, n_nodes, dataset_used, elapsed, attr, metrics_df, length)
            del clf, hist
            gc.collect()

    metrics_df.to_excel(os.path.join("Results", "metrics_First_Improvement_Relabeling.xlsx"))

def run_best_improvement_relabeling():
    metrics_df = pd.DataFrame()
    datasets = ["adult", "german", "compas", "bank"]
    protected_attributes = [["sex", "race"], ["sex", "age"], ["race", "sex"], ["age"]]
    grafic_df = pd.DataFrame(columns=["DataSet", "Atribute", "Operator", "Fairness", "Time"])
    max_time = [-1]
    data_tuple = (grafic_df, max_time)

    for dataset_index, dataset_used in enumerate(datasets):
        for attr in protected_attributes[dataset_index]:
            clf, train, test, valid, train_pred, test_pred, valid_pred, unpriv, priv, length = train_base_tree(dataset_used, attr)
            operator = "Best_improvement"
            hist = []
            start = time.time()
            clf, data_tuple = best_improvement_relabeling(clf, valid, valid_pred, unpriv, priv, hist, data_tuple, dataset_used, attr)
            elapsed = time.time() - start
            n_nodes = len(clf.tree_.children_left)
            metrics_df = write_metrics(clf, train, train_pred, unpriv, priv, test, test_pred, valid, valid_pred, operator, hist, n_nodes, dataset_used, elapsed, attr, metrics_df, length)
            del clf, hist
            gc.collect()

    metrics_df.to_excel(os.path.join("Results", "metrics_Best_Improvement_Relabeling.xlsx"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None,
                        choices=[
                            "state_of_art",
                            "first_prune", "best_prune",
                            "first_relabel", "best_relabel"])
    args = parser.parse_args()

    methods = {
        "state_of_art": run_state_of_art,
        "first_prune": run_first_improvement_prune,
        "best_prune": run_best_improvement_prune,
        "first_relabel": run_first_improvement_relabeling,
        "best_relabel": run_best_improvement_relabeling
    }

    if args.method is None:
        for func in methods.values():
            func()
    else:
        methods[args.method]()

if __name__ == "__main__":
    main()
