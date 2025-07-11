import os
import gc #importamos el recolector de basura

from tensorflow.python.ops.gen_stateless_random_ops import stateless_random_uniform_full_int

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
from utility import get_data
from sklearn import tree
import copy
import time
from funciones_dt_prune import get_metrics, best_improvement_prune, first_improvement_prune, get_state_of_art_algorithm
from funciones_dt_relabeling import first_improvement_relabeling, best_improvement_relabeling
from funciones_df import table_align, write_metrics

METHODS = ["state_of_art","first_prune","best_prune","first_relabel","best_relabel"]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="compas",
                    help="Dataset name")
parser.add_argument("-p", "--protected", type=str, default="race",
                    help="Protected attribute")

parser.add_argument("-s", "--start", type=int, default=0,
                    help="Start")
parser.add_argument("-e", "--end", type=int, default=50,
                    help="End")
parser.add_argument("-t", "--trials", type=int, default=50,
                    help="Trials")
parser.add_argument("-o", "--operations", type=int, default=2500,
                    help="Operations")
parser.add_argument("-m", "--metric", type=int, default=0,
                    help="metric")
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument("--method", type=str, nargs='+',choices=METHODS,default=METHODS, help="Método a ejecutar. Por defecto: TODOS")

args = parser.parse_args()
selected_methods = args.method


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


#Creamos los DF
columnas = ["Tree Hash","Operator", "Dataset Used","Lenght Dataset" ,"Attribute", "Train Accuracy", "Train AOD", "Test Accuracy", "Test AOD", "Validation Accuracy",
            "Validation AOD","Num Nodes", "Prune Count", "Elapsed Time (s)"]
metrics_prune_df = pd.DataFrame(columns=columnas)
metrics_relabeling_df = pd.DataFrame(columns=columnas)

columnas= ["DataSet","Atribute", "Operator", "Fairness", "Time"]
grafic_df = pd.DataFrame(columns=columnas)

max_time = [-1]

data_tuple = (grafic_df, max_time)

#Definimos diccionario de fairness para tener los valores iniciales de cada uno.
first_fairness = {}

#Comentar esto para una única ejecución y quitar identados
for dataset_index, dataset_used in enumerate(datasets):
    for attr in protected_attributes[dataset_index]:

        dataset_orig, privileged_groups,unprivileged_groups,optim_options = get_data(dataset_used, attr)
        lenght_data = dataset_orig.features.shape[0]

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
                      "No operator",  [[0]], 0, dataset_used, 0, attr,metrics_prune_df, lenght_data)
        metrics_relabeling_df = write_metrics(clf, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups,
                                         privileged_groups,dataset_orig_test, dataset_orig_test_pred, dataset_orig_valid,
                                         dataset_orig_valid_pred,"No operator", [[0]], 0, dataset_used, 0, attr, metrics_relabeling_df,lenght_data)


        #Guardamos el first fairness en el diccionario
        first_fairness[(dataset_used,attr)] = valid_fair


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
        if "state_of_art" in selected_methods:
            print("ejecutando state_of_art")
            start_time = time.time()
            clf, data_tuple = get_state_of_art_algorithm(clf,2500, n_nodes, prune_count,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,
                                                         privileged_groups, hist_art,data_tuple, dataset_used, attr)
            elapsed_time = time.time() - start_time
            n_nodes = len(clf.tree_.children_left)
            metrics_prune_df = write_metrics(clf, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,dataset_orig_test,
                          dataset_orig_test_pred, dataset_orig_valid,dataset_orig_valid_pred,operator[0],hist_art, n_nodes,
                          dataset_used, elapsed_time,attr, metrics_prune_df,lenght_data)

            #Borramos info del primer metodo para no sobrecargar:
        del clf, hist_art
        gc.collect()

        #Pruning, first improvement
        if "first_prune" in selected_methods:
            print("ejecutando first_prune")
            start_time = time.time()
            clf_first_prune, data_tuple = first_improvement_prune(clf_first_prune,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,
                                                                  privileged_groups, hist_first_prune,data_tuple,dataset_used, attr,0,None)
            elapsed_time = time.time() - start_time
            n_nodes = len(clf_first_prune.tree_.children_left)
            metrics_prune_df = write_metrics(clf_first_prune, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,dataset_orig_test,
                          dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, operator[1],hist_first_prune, n_nodes,
                          dataset_used,elapsed_time,attr, metrics_prune_df,lenght_data)

        #Relabeling, first improvement
        if "first_relabel" in selected_methods:
            print("ejecutando first_relabel")
            start_time = time.time()
            clf_first_relabeling, data_tuple= first_improvement_relabeling(clf_first_relabeling, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,
                                          privileged_groups, hist_first_relabeling,data_tuple, dataset_used, attr,0,None)
            elapsed_time = time.time() - start_time
            metrics_relabeling_df = write_metrics(clf_first_relabeling, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups,
                                       privileged_groups, dataset_orig_test,dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, operator[1],
                                       hist_first_relabeling, n_nodes,dataset_used, elapsed_time, attr, metrics_relabeling_df,lenght_data)


        #Borramos info del segundo metodo
        del clf_first_relabeling,hist_first_relabeling,clf_first_prune,hist_first_prune
        gc.collect()

        #Pruning, best improvement
        if "best_prune" in selected_methods:
            print("ejecutando best_prune")
            start_time = time.time()
            clf_best_prune,data_tuple = best_improvement_prune(clf_best_prune,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,
                                                               privileged_groups, hist_best_prune,data_tuple, dataset_used,attr)
            elapsed_time = time.time() - start_time
            n_nodes = len(clf_best_prune.tree_.children_left)
            metrics_prune_df = write_metrics(clf_best_prune, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,dataset_orig_test,
                          dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, operator[2], hist_best_prune, n_nodes,
                          dataset_used,elapsed_time, attr, metrics_prune_df,lenght_data)

        #Relabeling, best improvement
        if "best_relabel" in selected_methods:
            print("ejecutando best_relabel")
            start_time = time.time()
            clf_best_relabeling, data_tuple = best_improvement_relabeling(clf_best_relabeling, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,
                                                                          privileged_groups, hist_best_relabeling,data_tuple, dataset_used,attr)
            elapsed_time = time.time() - start_time
            metrics_relabeling_df = write_metrics(clf_best_relabeling, dataset_orig_train, dataset_orig_train_pred,unprivileged_groups, privileged_groups, dataset_orig_test,
                                             dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred,operator[2], hist_best_relabeling, n_nodes,
                                             dataset_used, elapsed_time, attr, metrics_relabeling_df,lenght_data)

        # Borramos info del tercer metodo
        del clf_best_prune, hist_best_prune, clf_best_relabeling, hist_best_relabeling
        gc.collect()


#Guardamos el Excel
if set(selected_methods) == set(METHODS):
    sufix = ""
else:
    sufix = "_" + "_".join(selected_methods)

excel_path = os.path.join("Results", f"metrics_Prune_results{sufix}.xlsx")
metrics_prune_df.to_excel(excel_path)

excel_path = os.path.join("Results", f"metrics_Relabeling_results{sufix}.xlsx")
metrics_relabeling_df.to_excel(excel_path)

prueba_df, invalid = data_tuple
excel_path = os.path.join("Results", f"prueba{sufix}.xlsx")
prueba_df.to_excel(excel_path)

grafic_df = table_align(data_tuple, first_fairness)

csv_path = os.path.join("Results", f"backup{sufix}.csv")
grafic_df.to_csv(excel_path, index=False)

excel_path = os.path.join("Results", f"grafic_metrics{sufix}.xlsx")
grafic_df.to_excel(excel_path)



