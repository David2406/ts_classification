import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.plot import plot_roc_curve
from src.rocket_doe import rocket_ks_dilations
from src.statistics import roc_curves_info
from src.utils import one_vs_rest_label
from src.convolution import RocketFeatures
from src.rocket_classifier import RocketClassifier

# In this script: we implement a basic version of the algorithm described in the article
# "ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels"
# Authors : Angus Dempster, François Petitjean, Geoffrey I. Webb
# Journal : Data Mining and Knowledge Discovery
# Year    : 2020

###############################################################################
##################### DATA LOADING & PREPROCESSING ############################
###############################################################################

save_results_folder_path = '/home/david/Dev/volta/saved_results/'
save_data_folder_path = '/home/david/Dev/volta/saved_data/'

patient_avg_ts = torch.load(os.path.join(save_data_folder_path,
                                         'patient_average_embeddings.pt'),
                            weights_only = True)

nb_pts = 5000
nb_leads = 12
nb_patients = 200
X = torch.reshape(patient_avg_ts, (nb_leads, nb_pts, nb_patients))
X = torch.transpose(X, 0, 2)
X = torch.transpose(X, 1, 2)

torch.save(X, os.path.join(save_data_folder_path, 'patient_average_ts.pt'))

###############################################################################
######################## BUILDING LABELS TENSOR ###############################
###############################################################################

ECG_COL_MAP = pd.read_pickle(os.path.join(save_data_folder_path, 'ECG_COL_MAP.pkl'))
ECG_COL_MAP['sinus_dummy'] = ECG_COL_MAP['rhy_grp'].apply(one_vs_rest_label)

sinus_dummy = ECG_COL_MAP[['id', 'sinus_dummy']].drop_duplicates()['sinus_dummy'].to_numpy()
other_dummy = 1. - sinus_dummy
y = np.stack([other_dummy, sinus_dummy], axis = 1)

####################################################################################
######################## ROCKET FEATURES GENERATIONS ###############################
####################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

nb_ks = 1000
ks_min = 3
ks_max = int(nb_pts / 2)

kernel_sizes_file_path = os.path.join(save_data_folder_path, 'rocket_kernel_sizes.pt')
dilations_file_path = os.path.join(save_data_folder_path, 'rocket_dilations.pt')

kernel_sizes, dilations = rocket_ks_dilations(ts_length = nb_pts,
                                              nb_ks = nb_ks,
                                              ks_bounds = [ks_min, ks_max],
                                              save_outputs = False,
                                              kernel_sizes_file_path = kernel_sizes_file_path,
                                              dilations_file_path = dilations_file_path)

rfeats_builder = RocketFeatures(kernel_sizes = kernel_sizes,
                                dilations = dilations,
                                in_channels = nb_leads,
                                out_channels = 1,
                                padding = 'same',
                                stride = 1,
                                weight_init_bounds = [-1., 1.],
                                bias_init_params = [0., 1.])

rocket_features_file_path = os.path.join(save_data_folder_path, 'rocket_features.pt')
rfeats = rfeats_builder.build_features(X,
                                       print_freq = 100,
                                       save_freq = 20,
                                       save_file_path = rocket_features_file_path)

# Remark: on my computer, without code optimization / CPU parallelization, I experienced that building the rocket features for nb_ks = 1000
# is computationally expensive even for only 200 patients. After a few hours, I ended up with nb_ks = 701 computed kernels, hence 1402 rocket features.

rfeats = torch.load(rocket_features_file_path)
kernel_sizes = torch.load(kernel_sizes_file_path)
dilations = torch.load(dilations_file_path)
ker_cut = 701
norm_rfeats = torch.nn.BatchNorm1d(2 * ker_cut, eps = 1e-05, momentum = 0.1, affine = False)(rfeats)

# Remark: I experienced that the range of rocket features (ppv and max of convolved kernels accross time series) can be quite large and can be
# different for one kernel and for an other. Hence, I decided to renormalize each rocket feature

norm_rocket_features_file_path = os.path.join(save_data_folder_path, 'norm_rocket_features.pt')
torch.save(norm_rfeats, norm_rocket_features_file_path)

kernel_sizes = kernel_sizes[:ker_cut]
dilations = dilations[:ker_cut]

#######################################################################
##################### DATASETS PREPARATION ############################
#######################################################################

norm_rfeats = norm_rfeats.numpy()
test_prop = 0.33
random_state = 1
shuffle = True

X_train, X_test, y_train, y_test = train_test_split(norm_rfeats,
                                                    y,
                                                    test_size = test_prop,
                                                    random_state = random_state,
                                                    shuffle = shuffle,
                                                    stratify = y)


#############################################################################
##################### ROCKET CLASSIFIER TRAINING ############################
#############################################################################

rocket_classifier = RocketClassifier(kernel_sizes = kernel_sizes,
                                     dilations = dilations,
                                     n_labels = 2,
                                     label_weights = None)

train_losses = rocket_classifier.fit(y_train = torch.from_numpy(y_train),
                                     rX_train = torch.from_numpy(X_train),
                                     loss_print_freq = 1000,
                                     epochs = 10000)

p_test = rocket_classifier.predict(rX_test = torch.from_numpy(X_test))


preds_data = np.concatenate([y_test, p_test.detach().numpy()], axis = 1)
label_names = ['other', 'sinus']
label_colors = ['red', 'blue']
score_names = ['p_other', 'p_sinus']
P_TEST = pd.DataFrame(data = preds_data, columns = label_names + score_names)

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
               save_filename = os.path.join(save_results_folder_path,
                                            'rocket_classifier',
                                            'rocket_classifier_roc.jpeg'))

# Remarks: for both classes AUC score is about 0.8 which is quite good performance knowing that there are only 200 patients and that
# only 701 kernels have been computed

# Ideas of improvement: in the Rocket approach the choice of the kernel characteristics is random. Besides, in the original article 10000 kernels are
# required to achieve a good accuracy (measured in terms of F1 score). If a time series has 5000 sampling points, it is rather odd to compute 20000 features
# in order to classify it.
# As an idea of improvement, I would better analyze the Fourier spectrum of the different lead time series to catch the frequencies (and thus the kernel sizes)
# that mostly appear in the sinus ECGs and not in the other ECGs and conversely. This would allow to considerably reduce the number of kernels to generate
# in the rocket approach
