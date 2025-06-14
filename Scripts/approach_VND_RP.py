import os
import gc #importamos el recolector de basura

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
        #sample_indices = np.random.choice(len(dataset_orig.features), 400, replace=False)
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



        #Array de operator para escribirlo en el documento:
        operator=["State_of_Art", "First_improvement","Best_improvement"]

        #array de históricos para cada operador
        hist = [(valid_acc,valid_fair,prune_count)]



        #Iniciamos el VND
        i =0
        global_time = time.time()
        while(i<2):
            # copias de arboles e hist para cada iteracion
            clf_prima = copy.deepcopy(clf)
            hist_copy = copy.deepcopy(hist)

            if(i==0):
                clf_prima, data_tuple = first_improvement_relabeling(clf_prima, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,
                                      privileged_groups, hist_copy,data_tuple, dataset_used, attr,1,global_time)
            elif (i==1):
                clf_prima,data_tuple = first_improvement_prune(clf_prima,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,
                                                              privileged_groups, hist_copy,data_tuple,dataset_used, attr,1,global_time)



            #miramos si mejora:
            valid_acc, valid_aod = get_metrics(clf_prima, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups)
            valid_fair = valid_aod
            prev_valid_acc, prev_valid_fair, p = hist[-1]
            if valid_fair < prev_valid_fair and valid_acc >= hist[0][0]:
                hist.append((valid_acc, valid_fair, prune_count))
                clf = copy.deepcopy(clf_prima)
                i = 0

            else:
                i = i + 1





#Guardamos el Excel

prueba_df, invalid = data_tuple
excel_path = os.path.join("Results", "results.xlsx")
prueba_df.to_excel(excel_path)




