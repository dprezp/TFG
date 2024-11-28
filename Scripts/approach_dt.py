import os
import gc #importamos el recolector de basura

from markdown_it.common.html_re import attribute

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import argparse
import random
import tqdm
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import StandardScaler

from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from utility import get_data,write_to_file
from sklearn.base import clone
from sklearn import tree
import copy
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import export_graphviz
import time
import graphviz
from funciones_dt_prune import get_metrics, write_metrics, best_improvement_prune, first_improvement_prune, get_state_of_art_algorithm
from funciones_dt_relabeling import first_improvement_relabeling, best_improvement_relabeling


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="compas",
                    help="Dataset name")
parser.add_argument("-p", "--protected", type=str, default="race",
                    help="Protected attribute")

parser.add_argument("-s", "--start", type=int, default=0,
                    help="Start")
parser.add_argument("-e", "--end", type=int, default=1,
                    help="End")
parser.add_argument("-t", "--trials", type=int, default=1,
                    help="Trials")
parser.add_argument("-o", "--operations", type=int, default=10,
                    help="Operations")
parser.add_argument("-m", "--metric", type=int, default=0,
                    help="metric")
parser.add_argument("--seed", type=int, default=1234, help="Random seed")


args = parser.parse_args()



#dataset_used = args.dataset # "adult", "german", "compas"
#attr = args.protected
start = args.start
end = args.end
trials = args.trials
operations = args.operations
metric_id = args.metric

# Definir datasets y sus atributos protegidos
datasets = ["adult", "german", "compas", "bank"]
protected_attributes = [["sex", "race"], ["sex", "age"], ["race", "sex"], ["age"]]  # Atributos protegidos por dataset
#dataset_used = "compas" #dataset de prueba
#attr = "race" #atributo  de prueba


#Creamos el DF
columnas = ["Tree Hash","Operator", "Dataset Used", "Attribute", "Train Accuracy", "Train AOD", "Test Accuracy", "Test AOD", "Validation Accuracy",
            "Validation AOD","Num Nodes", "Prune Count", "Elapsed Time (s)"]
metrics_prune_df = pd.DataFrame(columns=columnas)
metrics_relabeling_df = pd.DataFrame(columns=columnas)


#Comentar esto para una única ejecución y quitar identados
for dataset_index, dataset_used in enumerate(datasets):
    for attr in protected_attributes[dataset_index]:

        dataset_orig, privileged_groups,unprivileged_groups,optim_options = get_data(dataset_used, attr)

        #------------
        #Truncamos aleatoriamente a 1000 datos
        #sample_indices = np.random.choice(len(dataset_orig.features), 100, replace=False)
        #dataset_orig.features = dataset_orig.features[sample_indices]
        #dataset_orig.labels = dataset_orig.labels[sample_indices]
        #------------

        np.random.seed(1234)
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
        dataset_orig_test,dataset_orig_valid = dataset_orig_test.split([0.5], shuffle=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)


        # Creamos el árbol y lo entrenamos
        clf = tree.DecisionTreeClassifier(random_state=1)
        clf = clf.fit(dataset_orig_train.features, dataset_orig_train.labels)
        train_acc,train_aod = get_metrics(clf,dataset_orig_train,dataset_orig_train_pred,unprivileged_groups,privileged_groups)
        test_acc,test_aod = get_metrics(clf,dataset_orig_test,dataset_orig_test_pred,unprivileged_groups,privileged_groups)
        valid_acc,valid_aod = get_metrics(clf,dataset_orig_valid,dataset_orig_valid_pred,unprivileged_groups,privileged_groups)
        valid_fair = valid_aod
        metrics_prune_df = write_metrics(clf, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,
                      dataset_orig_test, dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred,
                      "No operator",  [[0]], 0, dataset_used, 0, attr,metrics_prune_df)
        metrics_relabeling_df = write_metrics(clf, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups,
                                         privileged_groups,dataset_orig_test, dataset_orig_test_pred, dataset_orig_valid,
                                         dataset_orig_valid_pred,"No operator", [[0]], 0, dataset_used, 0, attr, metrics_relabeling_df)

        #Sacamos métricas de nuestro árbol
        n_nodes = len(clf.tree_.children_left)
        leafs = [i for i,x in enumerate(clf.tree_.children_left) if x == -1]
        interior = [x for x in range(n_nodes) if x not in leafs]
        prune_count = 0

        #copias de arboles para cada operador
        clf_first_prune = copy.deepcopy(clf)
        clf_best_prune = copy.deepcopy(clf)
        clf_first_relabeling = copy.deepcopy(clf)
        clf_best_relabeling = copy.deepcopy(clf)

        #Array de operator para escribirlo en el documento:
        operator=["State_of_Art", "First_improvement","Best_improvement"]

        #array de históricos para cada operador
        hist_art = [(valid_acc,valid_fair,prune_count)]
        hist_first_prune = [(valid_acc, valid_fair, prune_count)]
        hist_best_prune = [(valid_acc, valid_fair, prune_count)]
        hist_first_relabeling = [(valid_acc, valid_fair, prune_count)]
        hist_best_relabeling = [(valid_acc, valid_fair, prune_count)]

        #Primer metodo del estado del arte
        start_time = time.time()
        clf = get_state_of_art_algorithm(clf,2500, n_nodes, prune_count,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist_art )
        elapsed_time = time.time() - start_time
        n_nodes = len(clf.tree_.children_left)
        metrics_prune_df = write_metrics(clf, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,dataset_orig_test,
                      dataset_orig_test_pred, dataset_orig_valid,dataset_orig_valid_pred,operator[0],hist_art, n_nodes,
                      dataset_used, elapsed_time,attr, metrics_prune_df)

        #Borramos info del primer metodo para no sobrecargar:
        del clf, hist_art
        gc.collect()

        #Pruning, first improvement
        start_time = time.time()
        clf_first = first_improvement_prune(clf_first_prune,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist_first_prune)
        elapsed_time = time.time() - start_time
        n_nodes = len(clf_first.tree_.children_left)
        metrics_prune_df = write_metrics(clf_first, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,dataset_orig_test,
                      dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, operator[1],hist_first_prune, n_nodes,
                      dataset_used,elapsed_time,attr, metrics_prune_df)

        #Relabeling, first improvement
        start_time = time.time()
        clf_first = first_improvement_relabeling(clf_first, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,
                                      privileged_groups, hist_first_relabeling)
        elapsed_time = time.time() - start_time
        metrics_relabeling_df = write_metrics(clf_first, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups,
                                   privileged_groups, dataset_orig_test,dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, operator[1],
                                   hist_first_relabeling, n_nodes,dataset_used, elapsed_time, attr, metrics_relabeling_df)


        #Borramos info del segundo metodo
        del clf_first_relabeling,hist_first_relabeling,clf_first,hist_first_prune
        gc.collect()

        #Pruning, best improvement
        start_time = time.time()
        clf_best_prune = best_improvement_prune(clf_best_prune,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist_best_prune)
        elapsed_time = time.time() - start_time
        n_nodes = len(clf_best_prune.tree_.children_left)
        metrics_prune_df = write_metrics(clf_best_prune, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,dataset_orig_test,
                      dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, operator[2], hist_best_prune, n_nodes,
                      dataset_used,elapsed_time, attr, metrics_prune_df)

        #Relabeling, best improvement
        # Tercer metodo, best improvement
        start_time = time.time()
        clf_best_relabeling = best_improvement_relabeling(clf_best_relabeling, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist_best_relabeling)
        elapsed_time = time.time() - start_time
        metrics_relabeling_df = write_metrics(clf_best_relabeling, dataset_orig_train, dataset_orig_train_pred,unprivileged_groups, privileged_groups, dataset_orig_test,
                                         dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred,operator[2], hist_best_relabeling, n_nodes,
                                         dataset_used, elapsed_time, attr, metrics_relabeling_df)

        # Borramos info del tercer metodo
        del clf_best_prune, hist_best_prune,
        gc.collect()

#Guardamos el Excel
excel_path = os.path.join("Results", "metrics_Prune_results.xlsx")
metrics_prune_df.to_excel(excel_path)

excel_path = os.path.join("Results", "metrics_Relabeling_results.xlsx")
metrics_relabeling_df.to_excel(excel_path)