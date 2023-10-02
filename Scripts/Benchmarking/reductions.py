import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from aif360.sklearn.inprocessing import ExponentiatedGradientReduction as skExpGradRed
from sklearn.preprocessing import StandardScaler
from utility import get_data,write_to_file,get_classifier
import os
import argparse
from sklearn import metrics

from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
#from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
#from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
#from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="adult",
                    help="Dataset name")
parser.add_argument("-c", "--clf", type=str, default="lr",
                    help="Classifier name")
parser.add_argument("-p", "--protected", type=str, default="sex",
                    help="Protected attribute")

parser.add_argument("-s", "--start", type=int, default=0,
                    help="Start")
parser.add_argument("-e", "--end", type=int, default=50,
                    help="End")
args = parser.parse_args()

scaler = StandardScaler()
dataset_used = args.dataset # "adult", "german", "compas"
attr = args.protected
clf_name = args.clf
start = args.start
end = args.end

# types = ["DemographicParity","EqualizedOdds","TruePositiveRateParity"]
constraint = "DemographicParity"
val_name = "reductions-{}_{}_{}_{}_{}_{}.txt".format(constraint,clf_name,dataset_used,attr,start,end)
val_name= os.path.join("reductions",val_name)

dataset_orig, privileged_groups,unprivileged_groups,optim_options = get_data(dataset_used, attr,preprocessed = True)



hist = []
for r in range(start,end):
    print (r)
    np.random.seed(r)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_test,dataset_orig_valid = dataset_orig_test.split([0.5], shuffle=True)
    #dataset_orig_train.features = scaler.fit_transform(dataset_orig_train.features)
    #dataset_orig_test.features = scaler.transform(dataset_orig_test.features)

    
    clf = get_classifier(clf_name)
    exp_grad_red = ExponentiatedGradientReduction(estimator=clf, 
                                              constraints=constraint,
                                              drop_prot_attr=False)
    exp_grad_red.fit(dataset_orig_train)
    exp_grad_red_pred = exp_grad_red.predict(dataset_orig_test)
    #clf = clf.fit(dataset_transf_train.features, dataset_transf_train.labels)
    #pred = clf.predict(dataset_orig_test.features).reshape(-1,1)

    #dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    #dataset_orig_test_pred.labels = pred

    
    class_metric = ClassificationMetric(dataset_orig_test, exp_grad_red_pred,
                     unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    
    stat = abs(class_metric.statistical_parity_difference())
    #aod = abs(class_metric.average_odds_difference())
    aod = abs(class_metric.average_abs_odds_difference())
    eod = abs(class_metric.equal_opportunity_difference())
    #print (size,abs(stat),accuracy)
    fpr, tpr, thresholds = metrics.roc_curve(dataset_orig_test.labels, exp_grad_red_pred.labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print (stat,class_metric.accuracy(),aod,eod,class_metric.precision(),class_metric.recall(),auc)
    content = "{} {} {} {} {} {} {}".format(stat,class_metric.accuracy(),aod,eod,class_metric.precision(),class_metric.recall(),auc)
    write_to_file(val_name,content)
    hist.append([stat,class_metric.accuracy(),aod,eod,class_metric.precision(),class_metric.recall(),auc])
    break
a = np.array(hist)
print (np.mean(a,axis=0))