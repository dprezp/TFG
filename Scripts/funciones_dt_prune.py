import gc
import os
import sys
import pandas as pd
from aif360.metrics import ClassificationMetric


import time
import numpy as np
import copy
import random
import hashlib
from sklearn.tree._tree import TREE_LEAF

from funciones_df import *






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





def get_state_of_art_algorithm (clf,operations,n_nodes, prune_count,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist, data_tuple,dataset_used, atribute):
    start_time = time.time()
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
        if valid_fair < prev_valid_fair and valid_acc >= prev_valid_acc:
            elapsed_time = time.time() - start_time
            prune_count += valor
            clf = copy.deepcopy(c)
            hist.append((valid_acc, valid_fair, prune_count))
            data_tuple = get_grafics(data_tuple,valid_fair,"State of art", elapsed_time,dataset_used, atribute )
    return clf, data_tuple

def first_improvement_prune (clf,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist, data_tuple, dataset_used, atribute,vnd,global_time):
    if(vnd==1):
        time_start = global_time
    else:
        time_start = time.time()
    mejora = True
    prune_count = 0
    while (mejora):
        mejora = False
        n_nodes = len(clf.tree_.children_left)
        leafs = [i for i, x in enumerate(clf.tree_.children_left) if x == -1]
        interior = [x for x in range(n_nodes) if x not in leafs]  # Coge el los interiores por cada operación, el nodo que ha sido prunned ya no es interior.

        while True:
            c = copy.deepcopy(clf)
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
            if valid_fair < prev_valid_fair and valid_acc >= hist[0][0]:
                elapsed_time = time.time() - time_start
                prune_count += valor
                clf = copy.deepcopy(c)
                hist.append((valid_acc, valid_fair, prune_count))
                mejora = True
                data_tuple = get_grafics(data_tuple,valid_fair,"First improvement pruning", elapsed_time, dataset_used, atribute)
                break
            gc.collect()
        gc.collect()

    return clf, data_tuple



def best_improvement_prune (clf,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist, data_tuple, dataset_used, atribute ):
    start_time = time.time()
    mejora = True
    prune_count = 0
    while (mejora):
        mejora = False
        n_nodes = len(clf.tree_.children_left)
        leafs = [i for i, x in enumerate(clf.tree_.children_left) if x == -1]
        interior = [x for x in range(n_nodes) if x not in leafs]  # Coge el los interiores por cada operación, el nodo que ha sido prunned ya no es interior.

        while True:
            c = copy.deepcopy(clf)
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
            if valid_fair < prev_valid_fair and valid_acc >= hist[0][0]:
                elapsed_time = time.time() - start_time
                prune_count += valor
                clf = copy.deepcopy(c)
                hist.append((valid_acc, valid_fair, prune_count))
                mejora = True
                data_tuple = get_grafics(data_tuple,valid_fair,"Best improvement pruning",elapsed_time, dataset_used, atribute)

            gc.collect()
        gc.collect()
    return clf, data_tuple
