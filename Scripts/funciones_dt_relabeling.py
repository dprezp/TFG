import gc
import numpy as np
import copy
import time

from funciones_dt_prune import get_metrics
from funciones_df import get_grafics


def relabel_leaf(clf, leaf_index, new_value):
    n_classes = clf.tree_.value.shape[2]
    new = np.zeros((1,n_classes))
    new[0, new_value] = 1
    clf.tree_.value[leaf_index]= new

    return clf

def first_improvement_relabeling (clf,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist, data_tuple, dataset_used, atribute):
    start_time = time.time()
    mejora = True
    tolerance = 0.001

    while (mejora):
        mejora = False
        leafs = [i for i, x in enumerate(clf.tree_.children_left) if x == -1]

        for leaf in leafs:
            c = copy.deepcopy(clf)

            original_value = c.tree_.value[leaf].ravel()
            new_value = 1 if np.argmax(original_value) == 0 else 0

            c = relabel_leaf(c, leaf, new_value)


            valid_acc, valid_aod = get_metrics(c, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,privileged_groups)
            valid_fair = valid_aod
            prev_valid_acc, prev_valid_fair, p = hist[-1]
            if valid_fair < prev_valid_fair and valid_acc >= hist[0][0] and abs(valid_fair - prev_valid_fair) > tolerance:
                elapsed_time = time.time() - start_time
                clf = copy.deepcopy(c)
                hist.append((valid_acc, valid_fair, leaf))
                mejora = True
                data_tuple = get_grafics(data_tuple, valid_fair, "First improvement relabel", elapsed_time, dataset_used, atribute)
                break
            else: #Si no mejora revertimos
                new_class = np.argmax(original_value)
                relabel_leaf(c, leaf, new_class)

            gc.collect()
        gc.collect()
    return clf, data_tuple



def best_improvement_relabeling (clf,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist, data_tuple, dataset_used, atribute):
    start_time = time.time()
    mejora = True
    tolerance = 0.001
    while (mejora):
        mejora = False
        best_leaf = None
        best_tree = None


        leafs = [i for i, x in enumerate(clf.tree_.children_left) if x == -1]


        for leaf in leafs:
            c = copy.deepcopy(clf)

            original_value = c.tree_.value[leaf].ravel()
            new_value = 1 if np.argmax(original_value) == 0 else 0

            c = relabel_leaf(c, leaf, new_value)


            valid_acc, valid_aod = get_metrics(c, dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups,privileged_groups)
            valid_fair = valid_aod
            prev_valid_acc, prev_valid_fair, p = hist[-1]

            if valid_fair < prev_valid_fair and valid_acc >= hist[0][0] and abs(valid_fair - prev_valid_fair) > tolerance:
                elapsed_time = time.time() - start_time
                best_leaf = leaf
                hist.append((valid_acc, valid_fair, leaf))
                best_tree = copy.deepcopy(c)
                data_tuple = get_grafics(data_tuple, valid_fair, "Best improvement relabel", elapsed_time, dataset_used, atribute)

            new_class = np.argmax(original_value)
            relabel_leaf(c, leaf, new_class)
            gc.collect()

        if best_leaf is not None:
            clf = copy.deepcopy(best_tree)
            mejora = True

        gc.collect()
    return clf, data_tuple
