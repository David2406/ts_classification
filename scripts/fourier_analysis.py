import pandas as pd
import numpy as np
import os
from src.config import init_config_options
import matplotlib.pyplot as plt
from src.plot import plot_fourier_ecg, rhy_color_mapping, rhy_label_mapping
from src.fourier import comp_fourier_spectrums, lead_cut_frequencies, filter_wt_frequency, build_fourier_df, plot_fourier_freq_hist


###########################################################################
##################### CONFIG PARAMETER LOADING ############################
###########################################################################

conf, work_dir, save_dir = init_config_options(config_file_path = './configs/data_exploration.json')
save_data = conf['save_data']

###############################################################################
##################### DATA LOADING & PREPROCESSING ############################
###############################################################################

ECG_COL_MAP = pd.read_pickle(os.path.join(save_data, 'ECG_COL_MAP.pkl'))
lead_names = np.array(ECG_COL_MAP['lead'].unique())
PATIENTS = ECG_COL_MAP[['id', 'rhy_grp', 'Rhythms', 'Age', 'gAge']].drop_duplicates()
norm_recds = np.load(os.path.join(save_data, 'norm_recds.npy'))
time_ts = np.load(os.path.join(save_data, 'time_ts.npy'))

nb_pts = 5000
time_step = 1 / 500

ecg_freqs, ecg_ks, fft_recds, fft_modes_ampl = comp_fourier_spectrums(norm_recds,
                                                                      nb_sample_pts = nb_pts,
                                                                      time_step = time_step,
                                                                      freq_bounds = None,
                                                                      save_outputs = True,
                                                                      fourier_freqs_file_path = os.path.join(save_data,
                                                                                                             'ecg_freqs.npy'),
                                                                      fourier_ks_file_path = os.path.join(save_data,
                                                                                                          'ecg_ks.npy'),
                                                                      fourier_transform_file_path = os.path.join(save_data,
                                                                                                                 'fft_recds.npy'),
                                                                      fourier_modes_ampl_file_path = os.path.join(save_data,
                                                                                                                  'fft_modes_ampl.npy'))
del norm_recds

###################################################################
##################### FOURIER ANALYSIS ############################
###################################################################

save_filename_pattern = os.path.join(save_dir,
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

####################################################################################
##################### FOURIER MODE AMPLITUDE HISTOGRAMS ############################
####################################################################################

AGG_FOURIER = build_fourier_df(ecg_freqs,
                               fft_modes_ampl,
                               lead_names,
                               PATIENTS,
                               nb_leads = 12,
                               freq_grid_step = 10.,
                               save_outputs = False,
                               delete_tables = True,
                               fourier_info_file_path = os.path.join(save_data, 'FOURIER_INFOS.pkl'),
                               agg_fourier_file_path = os.path.join(save_data, 'AGG_FOURIER.pkl'))

save_filename = 'fourier_hist_{}.jpeg'

for lead_name in lead_names:
    plot_fourier_freq_hist(LEAD_AGG_FOURIER = AGG_FOURIER[AGG_FOURIER.lead == lead_name],
                           lead_var = 'lead',
                           width = 0.15,
                           save_plot = True,
                           show_plot = True,
                           save_filename = os.path.join(save_dir, save_filename.format(lead_name)))

# Remarks: Fourier frequencies histograms whose weight is proportional to the average mode amplitude (for a given frequency, a given rhythm and a given lead)
# Remarks: These plots are just an aggregated vision of the previous plots

##############################################################################
##################### FOURIER CUTTING FREQUENCIES ############################
##############################################################################

CUT_FREQS = lead_cut_frequencies(ECG_COL_MAP,
                                 time_step,
                                 ecg_freqs,
                                 fft_modes_ampl,
                                 ampl_factor = 0.05,
                                 save_outputs = True,
                                 cut_freqs_file_path = os.path.join(save_data,
                                                                    'CUT_FREQS.pkl'))

# Remarks: with an amplitude factor of 5%, the median size of the filtering moving average window is 11 points
# Remarks: with an amplitude factor of 10%, the median size of the filtering moving average window is 19 points

##########################################################

filt_recds = filter_wt_frequency(ECG_COL_MAP,
                                 CUT_FREQS,
                                 norm_recds,
                                 nb_sample_pts = nb_pts,
                                 save_outputs = True,
                                 filt_recds_file_path = os.path.join(save_data,
                                                                     'filt_norm_recds.pt'))

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
