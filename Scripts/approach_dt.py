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
args = parser.parse_args()



dataset_used = args.dataset # "adult", "german", "compas"
attr = args.protected
start = args.start
end = args.end
trials = args.trials
operations = args.operations
metric_id = args.metric


val_name = "final_{}_{}_{}_{}_{}.txt".format(dataset_used,attr,start,trials,metric_id)
val_name= os.path.join("results",val_name)

dataset_orig, privileged_groups,unprivileged_groups,optim_options = get_data(dataset_used, attr)

def get_metrics(clf,test,test_pred,unprivileged_groups,privileged_groups):
    pred = clf.predict(test.features).reshape(-1,1)
    #dataset_orig_test_pred = test.copy(deepcopy=True)
    test_pred.labels = pred
    class_metric = ClassificationMetric(test, test_pred,
                         unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    stat = abs(class_metric.statistical_parity_difference())
    aod = abs(class_metric.average_abs_odds_difference())
    eod = abs(class_metric.equal_opportunity_difference())
    acc = class_metric.accuracy()
    return acc,stat,aod,eod


def prune(tree,interior):
    index = random.choice(interior)
    delete_left = prune_index(tree, tree.tree_.children_left[index])
    delete_right = prune_index(tree, tree.tree_.children_right[index])
    tree.tree_.children_left[index] = TREE_LEAF
    tree.tree_.children_right[index] = TREE_LEAF
    return tree, (delete_left+delete_right)

def prune_index(tree, index):
    delete_left = 0
    delete_right = 0
    if tree.tree_.children_left[index] != TREE_LEAF:
        delete_left += prune_index(tree, tree.tree_.children_left[index])
        delete_right += prune_index(tree, tree.tree_.children_right[index])
    tree.tree_.children_left[index] = TREE_LEAF
    tree.tree_.children_right[index] = TREE_LEAF
    return 1+delete_left+delete_right



for t in range(start,trials):
    content = "Trial {}".format(t)
    write_to_file(val_name,content)
    np.random.seed(t)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_test,dataset_orig_valid = dataset_orig_test.split([0.5], shuffle=True)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    for r in range(0,50):
        clf = tree.DecisionTreeClassifier(random_state=1)
        clf = clf.fit(dataset_orig_train.features, dataset_orig_train.labels)
        train_acc,train_stat,train_aod,train_eod = get_metrics(clf,dataset_orig_train,dataset_orig_train_pred,unprivileged_groups,privileged_groups)
        test_acc,test_stat,test_aod,test_eod = get_metrics(clf,dataset_orig_test,dataset_orig_test_pred,unprivileged_groups,privileged_groups)
        valid_acc,valid_stat,valid_aod,valid_eod = get_metrics(clf,dataset_orig_valid,dataset_orig_valid_pred,unprivileged_groups,privileged_groups)
        valid_fair = [valid_stat,valid_aod,valid_eod][metric_id]

        n_nodes = len(clf.tree_.children_left)
        leafs = [i for i,x in enumerate(clf.tree_.children_left) if x == -1]
        interior = [x for x in range(n_nodes) if x not in leafs]
        prune_count = 0
        hist = [(valid_acc,valid_fair,prune_count)]
        for o in range(operations):
            pruned = 0
            c = copy.deepcopy(clf)

            if n_nodes-prune_count <= 1:
                break
            leafs = [i for i,x in enumerate(clf.tree_.children_left) if x == -1]
            interior = [x for x in range(n_nodes) if x not in leafs]
            if len(interior)<=1:
                break
            c,pruned = prune(c,interior)     

            valid_acc,valid_stat,valid_aod,valid_eod = get_metrics(c,dataset_orig_valid,dataset_orig_valid_pred,unprivileged_groups,privileged_groups)  
            valid_fair = [valid_stat,valid_aod,valid_eod][metric_id]    
            prev_valid_acc,prev_valid_fair,p = hist[-1]
            if valid_fair < prev_valid_fair and valid_acc > prev_valid_acc:
                prune_count+=pruned
                clf = c
                hist.append((valid_acc,valid_fair,prune_count))

        train_acc,train_stat,train_aod,train_eod = get_metrics(clf,dataset_orig_train,dataset_orig_train_pred,unprivileged_groups,privileged_groups)
        test_acc,test_stat,test_aod,test_eod = get_metrics(clf,dataset_orig_test,dataset_orig_test_pred,unprivileged_groups,privileged_groups)
        valid_acc,valid_stat,valid_aod,valid_eod = get_metrics(clf,dataset_orig_valid,dataset_orig_valid_pred,unprivileged_groups,privileged_groups)
        
        content = "Run {}".format(r)
        write_to_file(val_name,content)
        prune_count = hist[-1][-1]
        content = "Data {} {}".format(n_nodes,prune_count)
        write_to_file(val_name,content)

        content = "{} {} {} {}".format(train_acc,train_stat,train_aod,train_eod)
        write_to_file(val_name,content)
        content = "{} {} {} {}".format(test_acc,test_stat,test_aod,test_eod)
        write_to_file(val_name,content)
        content = "{} {} {} {}".format(valid_acc,valid_stat,valid_aod,valid_eod)
        write_to_file(val_name,content)

    