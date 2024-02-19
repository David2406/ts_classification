import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from src.plot import plot_fourier_ecg, rhy_color_mapping, rhy_label_mapping
from src.fourier import comp_fourier_spectrums, lead_cut_frequencies, filter_wt_frequency

###############################################################################
##################### DATA LOADING & PREPROCESSING ############################
###############################################################################

save_results_folder_path = '/home/david/Dev/volta/saved_results/'
save_data_folder_path = '/home/david/Dev/volta/saved_data/'

ECG_COL_MAP = pd.read_pickle(os.path.join(save_data_folder_path, 'ECG_COL_MAP.pkl'))
norm_recds = np.load(os.path.join(save_data_folder_path, 'norm_ecg.npy'))
time_ts = np.load(os.path.join(save_data_folder_path, 'time_ts.npy'))

nb_pts = 5000
time_step = 1 / 500

ecg_freqs, ecg_ks, fft_recds, fft_modes_ampl = comp_fourier_spectrums(norm_recds,
                                                                      nb_sample_pts = nb_pts,
                                                                      time_step = time_step,
                                                                      freq_bounds = None,
                                                                      save_outputs = False,
                                                                      fourier_freqs_file_path = os.path.join(save_data_folder_path,
                                                                                                             'ecg_freqs.npy'),
                                                                      fourier_ks_file_path = os.path.join(save_data_folder_path,
                                                                                                          'ecg_ks.npy'),
                                                                      fourier_transform_file_path = os.path.join(save_data_folder_path,
                                                                                                                 'fft_recds.npy'),
                                                                      fourier_modes_ampl_file_path = os.path.join(save_data_folder_path,
                                                                                                                  'fft_modes_ampl.npy'))

###################################################################
##################### FOURIER ANALYSIS ############################
###################################################################

save_filename_pattern = os.path.join(save_results_folder_path,
                                     'fourier_analysis',
                                     'fourier_spretum_lead_{}.jpeg')

for lead, ecg_info_df in ECG_COL_MAP.groupby(['lead']):
    fig, ax = plot_fourier_ecg(ECG_INFOS = ecg_info_df,
                               freqs = ecg_freqs,
                               fft_recds = fft_recds,
                               rhy_color_mapping = rhy_color_mapping,
                               rhy_label_mapping = rhy_label_mapping,
                               x_label = 'Fourier frequency (Hz)',
                               y_label = 'Fourier mode amplitude',
                               legend_loc = 'right',
                               save_plot = False,
                               return_axis = True,
                               show_plot = False)
    plt.title('Fourier spectum for lead: {}'.format(lead))
    plt.savefig(save_filename_pattern.format(lead))
    plt.close()

# Remarks: for all patients, for all leads, the highest-amplitude Fourier modes have a frequency below 100 Hz
# Remarks: for leads (avl, avr, i, v1) a mode, whose en amplitude is relatively small, seems to appear at a frequency of 150 Hz

# Idea for time series filtering strategy: for each type of lead, apply a moving average whose width only keeps the highest-amplitude Fourier modes
# The cutting frequency criterion is the following: consider all modes whose amplitude > factor x (maximal amplitude),
# the cutting frequency is the highest frequency that meets this criterion. This criterion has 1 hyperparameter "factor" which here is set to 5%.

CUT_FREQS = lead_cut_frequencies(ECG_COL_MAP,
                                 time_step,
                                 ecg_freqs,
                                 fft_modes_ampl,
                                 ampl_factor = 0.05,
                                 save_outputs = False,
                                 cut_freqs_file_path = os.path.join(save_data_folder_path,
                                                                    'CUT_FREQS.pkl'))

# Remarks: with an amplitude factor of 5%, the median size of the filtering moving average window is 11 points
# Remarks: with an amplitude factor of 10%, the median size of the filtering moving average window is 19 points

##########################################################

filt_recds = filter_wt_frequency(ECG_COL_MAP,
                                 CUT_FREQS,
                                 norm_recds,
                                 nb_sample_pts = nb_pts,
                                 save_outputs = False,
                                 filt_recds_file_path = os.path.join(save_data_folder_path,
                                                                     'patient_average_embeddings.pt'))

##############################################################################################################
#################### VISUALISING THE EFFECT OF 1D MOVING AVERAGE FILTERING ###################################
##############################################################################################################

recd_col_id = 2398

raw_ts = norm_recds[:, recd_col_id]
filt_ts = filt_recds[:, recd_col_id]

fig, ax = plt.subplots()
plt.plot(time_ts, raw_ts, '-', color = 'blue', label = 'raw ecg')
plt.plot(time_ts, filt_ts, '-', color = 'red', label = 'filtered ecg')

ax.legend()
ax.xaxis.grid(True, which = 'major')
ax.yaxis.grid(True, which = 'major')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalized Voltage')
plt.show()
plt.close()
