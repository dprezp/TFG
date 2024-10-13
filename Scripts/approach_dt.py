import os
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
from funciones_dt import get_metrics, write_metrics, best_improvement, first_improvement, get_state_of_art_algorithm


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
datasets = ["adult", "german", "compas", "bank", "meps19"]
protected_attributes = [["sex", "race"], ["sex", "age"], ["race", "sex"], ["age"], ["race"]]  # Atributos protegidos por dataset


for dataset_index, dataset_used in enumerate(datasets):
    for attr in protected_attributes[dataset_index]:

        val_name = "final_{}_{}_{}_{}_{}.txt".format(dataset_used,attr,start,trials,metric_id)
        val_name= os.path.join("Results",val_name)

        dataset_orig, privileged_groups,unprivileged_groups,optim_options = get_data(dataset_used, attr)


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


        #Sacamos métricas de nuestro árbol
        n_nodes = len(clf.tree_.children_left)
        leafs = [i for i,x in enumerate(clf.tree_.children_left) if x == -1]
        interior = [x for x in range(n_nodes) if x not in leafs]
        prune_count = 0

        #copias de arboles para cada operador
        clf_first = copy.deepcopy(clf)
        clf_best = copy.deepcopy(clf)

        #Array de operator para escribirlo en el documento:
        operator=["State_of_Art", "First_improvement","Best_improvement"]

        #array de históricos para cada operador
        hist_art = [(valid_acc,valid_fair,prune_count)]
        hist_first = [(valid_acc, valid_fair, prune_count)]
        hist_best = [(valid_acc, valid_fair, prune_count)]

        #Primer operador del estado del arte
        start_time = time.time()
        get_state_of_art_algorithm(clf,2500, n_nodes, prune_count,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist_art )
        elapsed_time = time.time() - start_time
        write_metrics(clf, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,dataset_orig_test, dataset_orig_test_pred, dataset_orig_valid,dataset_orig_valid_pred,operator[0],val_name, hist_art, n_nodes, dataset_used, elapsed_time)

        #Segundo operador, first improvement
        start_time = time.time()
        first_improvement(clf_first,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist_first)
        elapsed_time = time.time() - start_time
        write_metrics(clf_first, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,dataset_orig_test, dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, operator[1],val_name, hist_first, n_nodes, dataset_used,elapsed_time)

        #Tercer operador, best improvement
        start_time = time.time()
        best_improvement(clf_best,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist_best)
        elapsed_time = time.time() - start_time
        write_metrics(clf_best, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,dataset_orig_test, dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, operator[2],val_name, hist_best, n_nodes, dataset_used,elapsed_time)
    