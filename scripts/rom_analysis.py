import numpy as np
import os
import torch
from src.config import init_config_options
from src.reduced_basis import ReducedBasis
from src.plot_reduced_basis import plot_cumulated_variance, plot_rb_leads

###########################################################################
##################### CONFIG PARAMETER LOADING ############################
###########################################################################

conf, work_dir, save_dir = init_config_options(config_file_path = './configs/rom_analysis.json')
save_data = conf['save_data']

###############################################################################
##################### DATA LOADING & PREPROCESSING ############################
###############################################################################

nb_leads = 12
filt_norm_recds = torch.load(os.path.join(save_data,
                                          'filt_norm_recds.pt'),
                             weights_only = True).numpy()

filt_norm_recds = np.split(filt_norm_recds,
                           indices_or_sections = nb_leads,
                           axis = 1)

X = np.concatenate(filt_norm_recds, axis = 0)

time_ts = np.load(os.path.join(save_data, 'time_ts.npy'))

###############################################################################
##################### REDUCED ORDER MODEL ANALYSIS ############################
###############################################################################

# ROM main idea: so far, a patient is "encoded" with 12 (number of leads) times series of 5000 time steps = a vector of 60000 entries
# The question is: can we encode a patient with a lower dimension vector ?
# Consider s_i(l, t) the value of the lead "l" at time t for the patient "i"
# The reduced order model for s_i is: s_i(l, t) = \bar{s}(l, t) + \sum_{k = 1}^{K} \alpha_k(i) x \Phi_k(l, t) + eps
# The vector \Phi_k of size 12 x 5000 are called the "reduced basis" or "modes" of the ROM
# The vector \alpha_k(i) k \in [1, K] are called the "reduced coordinates" of the ROM for the patient i

# We expect to find a K << 12 x 5000 such that \bar{s}(l, t) + \sum_{k = 1}^{K} \alpha_k(i) x \Phi_k(l, t) is a good approximation of s_i(l, t)
# in a certain sense. If it is, then each patient can be encoded with the vector [\alpha_1(i), ..., \alpha_K(i)] whose size is K.

nb_patients = 200
nb_pts = 5000
nb_leads = 12

patient_rb = ReducedBasis.from_snapshots(snapshot_matrix = X)
nb_modes = patient_rb.number_of_modes(exp_variance = 0.9,
                                      nb_max = nb_patients)

plot_cumulated_variance(base = patient_rb,
                        number_of_modes = nb_patients,
                        x_label = 'Number of modes',
                        y_label = 'Cumulated variance',
                        x_ticks = None,
                        y_ticks = None,
                        legend_loc = 'lower right',
                        title = 'Cumulated variance related to the patient model reduction',
                        linestyle = '-',
                        plot_style = 'o',
                        grid = True,
                        fig_size = (10, 8),
                        save_filename = os.path.join(save_dir,
                                                     'rom_analysis',
                                                     'rom_cumulated_variance.jpeg'))

# Remarks: with K = 167 reduced basis modes, 90% of the variability of the t \rightarrow s_i(l, t) time series is captured
# Remarks: even if K << 5 x 12, the time series can not easily be embedded with this linear method. Indeed the cumulated variance plot
# displays a linear dependence between the cumulated variance and the number of modes

patient_reduced_coords = patient_rb.reduced_coordinates(field = X,
                                                        modes = nb_modes)

patient_rc_file_path = os.path.join(save_data, 'patient_reduced_coords.npy')
np.save(patient_rc_file_path, patient_reduced_coords)

# Saving the patient reduced coordinates for the "naive classifier" approach (see scripts/simple_classifer.py)

###################################################################################################
########################### PLOTTING THE EXTRACTED MODES ##########################################
###################################################################################################

Phi = patient_rb.base[:, :nb_modes]
mode_id = 9

plot_rb_leads(mode_id,
              Phi,
              time_ts,
              nb_pts = nb_pts,
              nb_leads = nb_leads)
