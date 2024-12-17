import gc
import numpy as np
import copy
from funciones_dt_prune import get_metrics, write_metrics, get_grafics



def relabel_leaf(clf, leaf_index, new_value):
    n_classes = clf.tree_.value.shape[2]
    new = np.zeros((1,n_classes))
    new[0, new_value] = 1
    clf.tree_.value[leaf_index]= new

    return clf

def first_improvement_relabeling (clf,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist, grafic_df ):
    mejora = True
    tolerance = 1e-2

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
                clf = copy.deepcopy(c)
                hist.append((valid_acc, valid_fair, leaf))
                mejora = True
                grafic_df = get_grafics(grafic_df, valid_acc, "First improvement relabel")
                break
            else: #Si no mejora revertimos
                new_class = np.argmax(original_value)
                relabel_leaf(c, leaf, new_class)

            gc.collect()
        gc.collect()
    return clf, grafic_df



def best_improvement_relabeling (clf,dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups, privileged_groups, hist, grafic_df ):
    mejora = True
    tolerance = 1e-2
    while (mejora):
        mejora = False
        best_leaf = None
        best = (0,float('inf'))
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
                best_leaf = leaf
                best = (valid_acc, valid_fair)
                best_tree = copy.deepcopy(c)
                grafic_df = get_grafics(grafic_df, valid_acc, "Best improvement relabel")

            new_class = np.argmax(original_value)
            relabel_leaf(c, leaf, new_class)
            gc.collect()
        if best_leaf is not None:
            hist.append((best[0],best[1],best_leaf))
            clf = copy.deepcopy(best_tree)
            mejora = True

        gc.collect()
    return clf, grafic_df
