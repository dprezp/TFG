import random
from .utility import get_metrics
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from sklearn.base import clone
from sklearn import tree
import copy
import numpy as np
from sklearn.tree._tree import TREE_LEAF


def prune(tree,interior):
	""" Prunes a random interior node of a Decision Tree.

	Parameters:
		tree (DT classifer) -- DT classification model to be optimizied
		interior (lst) -- LIst of ids of interior nodes
	Returns:
		tree (DT classifer) -- Tree with pruned interior node
		int -- Number of pruned nodes
	"""
	index = random.choice(interior)
	delete_left = prune_index(tree, tree.tree_.children_left[index])
	delete_right = prune_index(tree, tree.tree_.children_right[index])
	tree.tree_.children_left[index] = TREE_LEAF
	tree.tree_.children_right[index] = TREE_LEAF
	return tree, (delete_left+delete_right)

def prune_index(tree, index):
	""" Prune tree at given node index.

	Parameters:
		tree (DT classifer) -- DT classification model to be optimizied
		index (int) -- Id of node to be pruned
	Returns:
		int -- Number of pruned nodes
	"""

	delete_left = 0
	delete_right = 0
	# Only prune if node is an interior node
	if tree.tree_.children_left[index] != TREE_LEAF:
		delete_left += prune_index(tree, tree.tree_.children_left[index])
		delete_right += prune_index(tree, tree.tree_.children_right[index])
	# Set children to leaf nodes
	tree.tree_.children_left[index] = TREE_LEAF
	tree.tree_.children_right[index] = TREE_LEAF
	return 1+delete_left+delete_right



def optimize_dt(clf, dataset_orig_valid, dataset_orig_valid_pred,unprivileged_groups,privileged_groups,operations = 2500, fairness_metric = 0):
	""" Optimize decision trees.

	Parameters:
		clf (classifer) -- Classifier with default configuration from scipy
		dataset_orig_valid (array) -- Validation data
		dataset_orig_valid_pred (array) -- Predicted labels of validation data
		unprivileged_groups(list)--Attribute and label of unprivileged group
		privileged_groups (list) -- Attribute and label of privileged group
		operations (int) -- Number of optmization operations
		fairness_metric (int) -- ID of fairness metric to optimize for (0=SPD, 1=AOD, 2=EOD)
		
	Returns:
		clf (classifier) -- Modified classification model
	"""

	# Determine initial performance
	valid_acc,valid_stat,valid_aod,valid_eod = get_metrics(clf,dataset_orig_valid,dataset_orig_valid_pred,unprivileged_groups,privileged_groups)
	valid_fair = [valid_stat,valid_aod,valid_eod][fairness_metric]


	n_nodes = len(clf.tree_.children_left)
	leafs = [i for i,x in enumerate(clf.tree_.children_left) if x == -1]
	interior = [x for x in range(n_nodes) if x not in leafs]
	prune_count = 0
	hist = [(valid_acc,valid_fair,prune_count)]

	for o in range(operations):
		# Make a copy of the DT (needed for undoing of pruning)
		pruned = 0
		c = copy.deepcopy(clf)

		# Stop if only root node left
		if n_nodes-prune_count <= 1:
			break
		leafs = [i for i,x in enumerate(clf.tree_.children_left) if x == -1]
		interior = [x for x in range(n_nodes) if x not in leafs]
		if len(interior)<=1:
			break
		# Prune one of the interior nodes
		c,pruned = prune(c,interior)     

		# Determine if both accuracy and fairness improved
		valid_acc,valid_stat,valid_aod,valid_eod = get_metrics(c,dataset_orig_valid,dataset_orig_valid_pred,unprivileged_groups,privileged_groups)  
		valid_fair = [valid_stat,valid_aod,valid_eod][fairness_metric]    
		prev_valid_acc,prev_valid_fair,p = hist[-1]
		if valid_fair < prev_valid_fair and valid_acc > prev_valid_acc:
			prune_count+=pruned
			clf = c
		hist.append((valid_acc,valid_fair,prune_count))
	return clf
    
	
def optimize_lr(clf, dataset_orig_valid, dataset_orig_valid_pred,unprivileged_groups,privileged_groups,operations = 2500, fairness_metric = 0,noise=0.1):
	""" Optimize logistic regression models.

	Parameters:
		clf (classifer) -- Classifier with default configuration from scipy
		dataset_orig_valid (array) -- Validation data
		dataset_orig_valid_pred (array) -- Predicted labels of validation data
		unprivileged_groups(list)--Attribute and label of unprivileged group
		privileged_groups (list) -- Attribute and label of privileged group
		operations (int) -- Number of optmization operations
		fairness_metric (int) -- ID of fairness metric to optimize for (0=SPD, 1=AOD, 2=EOD)
		noise (float) -- Level of noise used for mutation
		
	Returns:
		clf (classifier) -- Modified classification model
	"""

	# Determine initial performance
	_,n = clf.coef_.shape
	valid_acc,valid_stat,valid_aod,valid_eod = get_metrics(clf,dataset_orig_valid,dataset_orig_valid_pred,unprivileged_groups,privileged_groups)
	valid_fair = [valid_stat,valid_aod,valid_eod][fairness_metric]
	hist = [(valid_acc,valid_fair)]
	for op in range(operations):
		# Pick random element to mutate
		i = random.randint(0,n)
		change = np.random.uniform(-noise,noise,1)[0]
		if i == n:
			clf.intercept_*=change
		else:
			clf.coef_[0][i] *=change
		# Determine if both accuracy and fairness improved
		valid_acc,valid_stat,valid_aod,valid_eod = get_metrics(clf,dataset_orig_valid,dataset_orig_valid_pred,unprivileged_groups,privileged_groups)  
		valid_fair = [valid_stat,valid_aod,valid_eod][fairness_metric]    
		prev_valid_acc,prev_valid_fair = hist[-1]
		if valid_fair < prev_valid_fair and valid_acc>prev_valid_acc:
			hist.append((valid_acc,valid_fair))
		else:
			# Undo changes
			rev_change = 1/change
			if n == i:
				clf.intercept_*=rev_change
			else:
				clf.coef_[0][i] *=rev_change
	return clf
