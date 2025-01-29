import os

import pandas as pd
from sklearn.tree import export_graphviz
import graphviz

from funciones_dt_prune import get_metrics, hash_decisiontree


def get_grafics(data_tuple, fairness, operator, time, dataset):
    grafic_df, max_time = data_tuple

    nueva_fila = {
        "DataSet": dataset,
        "Operator": operator,
        "Fairness": fairness,
        "Time": time
    }

    grafic_df.loc[len(grafic_df)] = nueva_fila

    if ( max_time[0] < time):
        max_time[0] = time

    return grafic_df , max_time

def write_metrics(clf, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,dataset_orig_test,
                  dataset_orig_test_pred, dataset_orig_valid,dataset_orig_valid_pred, operator,
                  hist, n_nodes, dataset_used, elapsed_time,attr, metrics_df):


    train_acc, train_aod = get_metrics(clf, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups)
    test_acc, test_aod = get_metrics(clf, dataset_orig_test, dataset_orig_test_pred, unprivileged_groups, privileged_groups)
    valid_acc, valid_aod = get_metrics(clf, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups)

    hash = hash_decisiontree(clf)

    metrics_dict = {
        "Tree Hash": hash,
        "Operator": operator,
        "Dataset Used": dataset_used,
        "Attribute": attr,
        "Train Accuracy": train_acc,
        "Train AOD": train_aod,
        "Test Accuracy": test_acc,
        "Test AOD": test_aod,
        "Validation Accuracy": valid_acc,
        "Validation AOD": valid_aod,
        "Num Nodes": n_nodes,
        "Prune Count" : hist[-1][-1],
        "Elapsed Time (s)": elapsed_time
    }

    metrics_row = pd.DataFrame([metrics_dict])
    metrics_df = pd.concat([metrics_df, metrics_row], axis=0, ignore_index = True)



    # Guardamos imagen del árbol generado
    os.environ["PATH"] += os.pathsep + r'C:\\Program Files\\Graphviz\\bin'
    # Exportar el árbol de decisión a formato .dot
    dot_data = export_graphviz(clf, out_file=None,
                               feature_names=dataset_orig_train.feature_names,
                               class_names=["Unprivileged", "Privileged"],
                               filled=False, rounded=True,
                               special_characters=False)

    # Usar graphviz para convertir el archivo .dot a un gráfico
    graph = graphviz.Source(dot_data)

    # Guardar el gráfico en formato PNG
    name = os.path.join("Results\\Images", "best_decision_tee_{}_{}_{}".format(operator,dataset_used,attr))
    graph.render(name, format="pdf")

    return metrics_df


def table_align (data_tuple):
    grafic_df, max_time = data_tuple

    #variable de tiempo
    tiempo_maximo = max_time[0]

    #nuevo df para corregir filas
    aligned_df = pd.DataFrame(columns=grafic_df.columns)

    #iteramos sobre el las combinaciones
    for (dataset, operator), group in grafic_df.groupby(['DataSet', 'Operator']):
        #ordenamos por tiempo
        group = group.sort_values('Time').reset_index(drop=True)

        #creamos el fairnesss para utilizarlo más tarde
        fairness_dict = {}
        last_fairness = 0

        for _, row in group.iterrows():
            time = row['Time']
            fairness_dict[time] = row['Fairness']

        for t in range(0, tiempo_maximo +1):
            if t in fairness_dict:
                last_fairness = fairness_dict[t]

            aligned_df = pd.concat([aligned_df,pd.DataFrame([{
                "DataSet": dataset,
                "Operator": operator,
                "Fairness": last_fairness,
                "Time": t
            }])], ignore_index=True)

    return aligned_df