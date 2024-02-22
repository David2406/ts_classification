import numpy as np
import pandas as pd
import os
from src.config import init_config_options
from src.utils import one_vs_rest_label
from src.plot import plot_roc_curve
from src.statistics import roc_curves_info, format_binary_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

###########################################################################
##################### CONFIG PARAMETER LOADING ############################
###########################################################################

conf, work_dir, save_dir = init_config_options(config_file_path = './configs/simple_classifier.json')
save_data = conf['save_data']

###############################################################################
##################### DATA LOADING & PREPROCESSING ############################
###############################################################################

X_t = np.load(os.path.join(save_data, 'patient_reduced_coords.npy'))
X = np.transpose(X_t)

ECG_COL_MAP = pd.read_pickle(os.path.join(save_data, 'ECG_COL_MAP.pkl'))
ECG_COL_MAP['sinus_dummy'] = ECG_COL_MAP['rhy_grp'].apply(one_vs_rest_label)
y = ECG_COL_MAP[['id', 'sinus_dummy']].drop_duplicates()['sinus_dummy'].to_numpy()

#######################################################################
##################### DATASETS PREPARATION ############################
#######################################################################

random_state = 1
shuffle = True
test_prop = 0.33
label_names = ['other', 'sinus']
label_colors = ['red', 'blue']
score_names = ['p_other', 'p_sinus']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = test_prop,
                                                    random_state = random_state,
                                                    shuffle = shuffle,
                                                    stratify = y)

# Remark: stratify = y to garantee that the same proportions of sinus/other classes are present in the training/test datasets


log_reg = LogisticRegression(penalty = None,
                             class_weight = 'balanced',
                             random_state = random_state).fit(X_train, y_train)

P_TEST = format_binary_classification(classifier = log_reg,
                                      X = X_test,
                                      y = y_test,
                                      label_names = label_names,
                                      pred_names = score_names)

###############################################################################################
##################### ROC CURVES & AUC SCORE AS METRICS OF PERFORMANCE ########################
###############################################################################################

fpr, tpr, thresholds, roc_auc = roc_curves_info(data = P_TEST,
                                                label_names = label_names,
                                                score_names = score_names)

plot_roc_curve(fpr,
               tpr,
               roc_auc,
               label_names,
               label_colors,
               save_plot = True,
               show_plot = True,
               save_filename = os.path.join(save_dir,
                                            'rom_analysis',
                                            'simple_classifier_roc.jpeg'))


# Remark: AUC score is around 1 / 2; this means that the simple classifier's predictions are almost equivalent to a random classifier
# 'other' or 'sinus' with a 50% probability

##################################################################################################################################################
# REMARKS: with the above approach (reduced coordinates + simple classifier) we do not generate features that invariant with time-translation
# REMARKS: with the above approach (reduced coordinates + simple classifier) we do not generate features that amplitude-scaling-invariant
# (individuals with different ages but the same rhythm)
# REMARKS: that is why, reduced coordinates are not discriminative to provide accurate predictions
##################################################################################################################################################
