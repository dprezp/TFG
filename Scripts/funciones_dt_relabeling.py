import gc

import graphviz
import os
import sys

import pandas as pd
from aif360.metrics import ClassificationMetric
from utility import get_data,write_to_file
from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree
import numpy as np
import copy
import random
import hashlib
from sklearn.tree._tree import TREE_LEAF
from funciones_dt_prune import get_metrics, write_metrics



def relabel_leaf(clf, leaf_index, new_value):
    clf.tree_.value[leaf_index] = new_value
    return clf

def first_improvement_relabeling (clf,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist ):
    mejora = True
    tolerance = 1e-6

    while (mejora):
        mejora = False
        c = copy.deepcopy(clf)
        leafs = [i for i, x in enumerate(clf.tree_.children_left) if x == -1]

        for leaf in leafs:
            original_value = c.tree_.value[leaf].copy()
            new_value = 1 -np.argmax(original_value)
            relabel_leaf(c, leaf, new_value)


            valid_acc, valid_aod = get_metrics(c, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,privileged_groups)
            valid_fair = valid_aod
            prev_valid_acc, prev_valid_fair, p = hist[-1]
            if valid_fair < prev_valid_fair and valid_acc >= prev_valid_acc and abs(valid_fair - prev_valid_fair) > tolerance and abs(valid_acc - prev_valid_acc) > tolerance:
                clf = copy.deepcopy(c)
                hist.append((valid_acc, valid_fair, leaf))
                mejora = True
                break
            else: #Si no mejora revertimos
                relabel_leaf(c, leaf, original_value)

            gc.collect()
        gc.collect()
    return clf



def best_improvement_relabeling (clf,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist ):
    mejora = True
    tolerance = 1e-6
    while (mejora):
        mejora = False
        best_leaf = None
        best = (0,float('inf'))
        best_tree = None

        c = copy.deepcopy(clf)
        leafs = [i for i, x in enumerate(clf.tree_.children_left) if x == -1]


        for leaf in leafs:
            original_value = c.tree_.value[leaf].copy()
            new_value = 1 -np.argmax(original_value)
            relabel_leaf(c, leaf, new_value)


            valid_acc, valid_aod = get_metrics(c, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,privileged_groups)
            valid_fair = valid_aod
            prev_valid_acc, prev_valid_fair, p = hist[-1]

            if valid_fair < prev_valid_fair and valid_acc >= prev_valid_acc and abs(valid_fair - prev_valid_fair) > tolerance and abs(valid_acc - prev_valid_acc) > tolerance:
                if valid_acc > best[0] or (valid_acc == best[0] and valid_fair < best[1]):
                    best_leaf = leaf
                    best = (valid_acc, valid_fair)
                    best_tree = copy.deepcopy(c)

            relabel_leaf(c, leaf, original_value)
            gc.collect()
        if best_leaf is not None:
            hist.append((best[0],best[1],best_leaf))
            clf = copy.deepcopy(best_tree)
            mejora = True

        gc.collect()
    return clf
