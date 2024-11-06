import gc

import graphviz
import os
import sys
from aif360.metrics import ClassificationMetric
from utility import get_data,write_to_file
from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree
import copy
import random
from sklearn.tree._tree import TREE_LEAF


def get_metrics(clf,test,test_pred,unprivileged_groups,privileged_groups):
    pred = clf.predict(test.features).reshape(-1,1)
    #dataset_orig_test_pred = test.copy(deepcopy=True)
    test_pred.labels = pred
    class_metric = ClassificationMetric(test, test_pred,
                         unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    aod = abs(class_metric.average_abs_odds_difference())
    acc = class_metric.accuracy()
    return acc,aod


def prune(tree,interior):
    index = random.choice(interior)
    delete_left = prune_index(tree, tree.tree_.children_left[index])
    delete_right = prune_index(tree, tree.tree_.children_right[index])
    tree.tree_.children_left[index] = TREE_LEAF
    tree.tree_.children_right[index] = TREE_LEAF
    return tree, index, (delete_left+delete_right)

def prune_index(tree, index):
    delete_left = 0
    delete_right = 0
    if tree.tree_.children_left[index] != TREE_LEAF:
        delete_left += prune_index(tree, tree.tree_.children_left[index])
        delete_right += prune_index(tree, tree.tree_.children_right[index])
    tree.tree_.children_left[index] = TREE_LEAF
    tree.tree_.children_right[index] = TREE_LEAF
    return 1+delete_left+delete_right




def write_metrics(clf, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups, privileged_groups,dataset_orig_test, dataset_orig_test_pred, dataset_orig_valid,dataset_orig_valid_pred, operator, val_name, hist, n_nodes, dataset_used, elapsed_time,attr):
    train_acc, train_aod = get_metrics(clf, dataset_orig_train, dataset_orig_train_pred, unprivileged_groups,
                                       privileged_groups)
    test_acc, test_aod = get_metrics(clf, dataset_orig_test, dataset_orig_test_pred, unprivileged_groups,
                                     privileged_groups)
    valid_acc, valid_aod = get_metrics(clf, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,
                                       privileged_groups)

    content = "Operator: {}".format(operator)
    write_to_file(val_name, content)
    prune_count = hist[-1][-1]
    content = "Data {} {}".format(n_nodes, prune_count)
    write_to_file(val_name, content)
    content = "{} {}".format(train_acc, train_aod)
    write_to_file(val_name, content)
    content = "{} {}".format(test_acc, test_aod)
    write_to_file(val_name, content)
    content = "{} {}".format(valid_acc, valid_aod)
    write_to_file(val_name, content)

    # Guardamos el tiempo de ejecución
    content = "Time taken for {}: {:.4f} seconds".format(operator, elapsed_time)
    write_to_file(val_name, content)


    # Guardamos imagen del árbol generado
    #os.environ["PATH"] += os.pathsep + r'C:\\Program Files\\Graphviz\\bin'
    # Exportar el árbol de decisión a formato .dot
    #dot_data = export_graphviz(clf, out_file=None,
    #                           feature_names=dataset_orig_train.feature_names,
    #                           class_names=["Unprivileged", "Privileged"],
    #                           filled=True, rounded=True,
    #                           special_characters=False)

    # Usar graphviz para convertir el archivo .dot a un gráfico
    #graph = graphviz.Source(dot_data)

    # Guardar el gráfico en formato PNG
    #name = os.path.join("Results\\Images", "best_decision_tee_{}_{}_{}".format(operator,dataset_used,attr))
    #graph.render(name, format="png")


def get_state_of_art_algorithm (clf,operations,n_nodes, prune_count,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist ):
    for o in range(operations):
        pruned = 0
        c = copy.deepcopy(clf)

        if n_nodes - prune_count <= 1:
            break
        leafs = [i for i, x in enumerate(clf.tree_.children_left) if x == -1]
        interior = [x for x in range(n_nodes) if
                    x not in leafs]  # Coge el los interiores por cada operación, el nodo que ha sido prunned ya no es interior.
        if len(interior) <= 1:
            break
        c, pruned, valor = prune(c, interior)

        valid_acc, valid_aod = get_metrics(c, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,privileged_groups)
        valid_fair = valid_aod
        prev_valid_acc, prev_valid_fair, p = hist[-1]
        if valid_fair < prev_valid_fair and valid_acc > prev_valid_acc:
            prune_count += valor
            clf = c
            hist.append((valid_acc, valid_fair, prune_count))

def first_improvement (clf,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist ):
    mejora = True
    prune_count = 0
    while (mejora):
        mejora = False
        c = copy.deepcopy(clf)
        n_nodes = len(clf.tree_.children_left)
        leafs = [i for i, x in enumerate(clf.tree_.children_left) if x == -1]
        interior = [x for x in range(n_nodes) if x not in leafs]  # Coge el los interiores por cada operación, el nodo que ha sido prunned ya no es interior.

        while True:

            if n_nodes - prune_count <= 1:
                break

            if len(interior) <= 1:
                break


            c, pruned, valor = prune(c, interior)
            if (pruned in interior):
                interior.remove(pruned)
                leafs.append(pruned)
            valid_acc, valid_aod = get_metrics(c, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,privileged_groups)
            valid_fair = valid_aod
            prev_valid_acc, prev_valid_fair, p = hist[-1]
            if valid_fair < prev_valid_fair and valid_acc > prev_valid_acc:
                prune_count += valor
                clf = c
                hist.append((valid_acc, valid_fair, prune_count))
                mejora = True
                break
            gc.collect()
        gc.collect()



def best_improvement(clf,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist):
    mejora = True
    prune_count = 0
    while (mejora):
        mejora = False
        pruned = 0
        c = copy.deepcopy(clf)
        n_nodes = len(clf.tree_.children_left)
        leafs = [i for i, x in enumerate(clf.tree_.children_left) if x == -1]
        interior = [x for x in range(n_nodes) if x not in leafs]  # Coge el los interiores por cada operación, el nodo que ha sido prunned ya no es interior.


        while True:

            if n_nodes - prune_count <= 1:
                break
            if len(interior) <= 1:
                break

            c, pruned, valor = prune(c, interior)
            if(pruned in interior):
                interior.remove(pruned)
                leafs.append(pruned)
            valid_acc, valid_aod = get_metrics(c, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups)
            valid_fair = valid_aod
            prev_valid_acc, prev_valid_fair, p = hist[-1]
            if valid_fair < prev_valid_fair and valid_acc > prev_valid_acc:
                prune_count += valor
                clf = c
                hist.append((valid_acc, valid_fair, prune_count))
                mejora = True
            gc.collect()
        gc.collect()

