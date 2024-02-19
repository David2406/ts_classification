import pandas as pd
import numpy as np
import pickle
import os
from src.preprocessing import load_ludb_records, load_patient_df, rhy_mapping, pivot_ecg_records, norm_ecgs
from src.plot import plot_ecg_ts, plot_by_group, plot_ecg_ts_by_grp
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
cmap = cm.get_cmap('Spectral')
from matplotlib import colors as mcolors

###############################################################################
##################### DATA LOADING & PREPROCESSING ############################
###############################################################################

save_results_folder_path = '/home/david/Dev/volta/saved_results/'
save_data_folder_path = '/home/david/Dev/volta/saved_data/'

RECDS = load_ludb_records()
PATIENTS = load_patient_df(rhy_mapping = rhy_mapping)

RECDS = pd.merge(RECDS,
                 PATIENTS[['id', 'Age', 'gAge', 'rhy_grp']],
                 on = ['id'],
                 how = 'left')

RECDS.to_pickle(os.path.join(save_data_folder_path, 'RECDS.pkl'))
PATIENTS.to_pickle(os.path.join(save_data_folder_path, 'PATIENTS.pkl'))

time_ts = RECDS[['time']].drop_duplicates().to_numpy()
np.save(os.path.join(save_data_folder_path, 'time_ts.npy'), time_ts)

P_RECDS, ECG_COL_MAP = pivot_ecg_records(RECDS,
                                         PATIENTS[['id', 'rhy_grp', 'Rhythms', 'Age', 'gAge']],
                                         save_outputs = False,
                                         pivot_recds_file_path = os.path.join(save_data_folder_path, 'P_RECDS.pkl'),
                                         ecg_col_map_file_path = os.path.join(save_data_folder_path, 'ECG_COL_MAP.pkl'))


norm_recds, ts_mean_recds, ts_std_recds = norm_ecgs(P_RECDS,
                                                    save_outputs = False,
                                                    norm_ecg_file_path = os.path.join(save_data_folder_path, 'norm_recds.npy'))


#####################################################################
##################### DATA VISUALIZATION ############################
#####################################################################

# Rhythms & Age histograms

n_bins = 20
x_label = 'Age'
y_label = 'Frequency'

save_filename = os.path.join(save_results_folder_path, 'visualization', 'age_hist.jpeg')
fig, ax = plt.subplots()
ax = PATIENTS.Age.plot(kind = 'hist',
                       stacked = True,
                       bins = n_bins,
                       grid = True)
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
plt.show()
fig.savefig(save_filename)
plt.close()

# Remarks: the median age of the patient dataset is 56 years old

save_filename = os.path.join(save_results_folder_path, 'visualization', 'age_rhy_hist.jpeg')
fig, ax = plt.subplots()
ax = PATIENTS[['Age', 'rhy_grp']].pivot(columns = 'rhy_grp').Age.plot(kind = 'hist',
                                                                      stacked = True,
                                                                      bins = n_bins,
                                                                      grid = True)
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
plt.show()
fig.savefig(save_filename)
plt.close()

# Remarks: bradycardia is fairly distributed among all ages
# Remarks: tachycardia seems to be present for patients < 56 years old
# Remarks: atrial fibrillation seems to be present for patients > 56 years old

##################### ECG PLOTS ##################################

# Plot of the 'i'-lead of all "Sinus Rhythms": the idea is to observe the potential dissimilarities between ECGS for different ages

SINUS = RECDS[RECDS.rhy_grp == 'sinus']
sinus_age_grps = np.sort(np.array(SINUS.gAge.unique()))
sinus_colors = [mcolors.rgb2hex(cmap(age / 100)) for age in sinus_age_grps]

x_var = 'time'
y_var = 'i'
x_ticks = np.arange(0., 10.2, 0.2)
y_ticks = np.arange(-0.6, 1.1, 0.1)
x_label = 'Time (s)'
y_label = 'Voltage of {} lead (V)'.format(y_var)
legend_loc = 'lower right'

x_lims = [2., 3.]
y_lims = None

fig, ax = plt.subplots()

seen_ages = []
for patient_id, patient_df in SINUS.groupby(['id']):

    grid_age = patient_df.iloc[0]['gAge']

    plt.plot(patient_df[x_var],
             patient_df[y_var],
             '-',
             color = mcolors.rgb2hex(cmap(grid_age / 100)),
             label = grid_age if not (grid_age in seen_ages) else None)

    seen_ages.append(grid_age)

ax.legend(loc = legend_loc)
ax.set_xticks(x_ticks, minor = False)
ax.xaxis.grid(True, which = 'major')
ax.set_yticks(y_ticks, minor = False)
ax.yaxis.grid(True, which = 'major')
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)

if not x_lims is None:
    plt.xlim(x_lims[0], x_lims[1])
if not y_lims is None:
    plt.ylim(y_lims[0], y_lims[1])

plt.xticks(rotation = 45)
plt.yticks()
plt.show()
plt.close()

# Remarks: even for a 1-second snapshot, it is not easy to catch many informations
# Remarks: for all people ages, ecg amplitude lies between [-0.5, 0.9]
# Remarks: people > 90 years old has a distinct ecg time series (flat parts)
# Remarks: young people (w.r.t 56 years old) tends to display lower frequency modes compared to older people


################ PLOT ALL Z-NORMALIZED SINUS ECG ###########################

# Plot of the normalized 'i'-lead of all "Sinus Rhythms": the idea is to observe the potential amplitude differences between ECGS for different ages

y_var = 'i'
x_ticks = np.arange(0., 10.2, 0.2)
y_ticks = np.arange(-10., 10., 1)
x_label = 'Time (s)'
y_label = 'Normalized Voltage of the lead {}'.format(y_var)
legend_loc = 'lower right'


x_lims = [2., 3.]
# x_lims = None

SINUS = ECG_COL_MAP[(ECG_COL_MAP.rhy_grp == 'sinus') & (ECG_COL_MAP.lead == y_var)]

plot_ecg_ts(ECG_INFOS = SINUS,
            ecg_ts = norm_recds,
            time_ts = time_ts,
            cmap = cmap,
            x_ticks = x_ticks,
            y_ticks = y_ticks,
            x_label = x_label,
            y_label = y_label,
            x_lims = x_lims,
            save_plot = False,
            return_axis = False,
            show_plot = True)

# Remarks: steep peaks on normalized 'i'-lead ecgs are observed for old people (> 56 years-old)

################################################################################
######## PLOTS 56-years-old SINUS ECG VS 56-years-old bradycardia  #############
################################################################################

abnormal_rhy_grp = 'brady'
age = 56

y_var = 'i'
x_label = 'Time (s)'
y_label = 'Normalized Voltage of {} lead'.format(y_var)
legend_loc = 'lower right'

INFO_SUBSET = ECG_COL_MAP[(ECG_COL_MAP.id.isin([4, 59])) & (ECG_COL_MAP.lead == 'i')]

plot_ecg_ts_by_grp(ECG_INFOS = INFO_SUBSET,
                   ecg_ts = norm_recds,
                   time_ts = time_ts,
                   group_var = 'rhy_grp',
                   group_colors = {'sinus': 'darkblue',
                                   abnormal_rhy_grp: 'red'},
                   group_labels = {'sinus': 'sinus, ' + str(age),
                                   abnormal_rhy_grp: abnormal_rhy_grp + ', ' + str(age)},
                   x_label = x_label,
                   y_label = y_label,
                   x_lims = None,
                   show_plot = True)

# Remarks: bradycardia has a longer period wrt sinus rhythm (based on the pattern including P-wave + QRS complex + T-wave)

########################################################################################
######## PLOTS 56-years-old SINUS ECG VS 57-years-old Atrial Fibrillation  #############
########################################################################################

abnormal_rhy_grp = 'af'
age = 56

INFO_SUBSET = ECG_COL_MAP[(ECG_COL_MAP.id.isin([163, 8])) & (ECG_COL_MAP.lead == 'i')]

plot_ecg_ts_by_grp(ECG_INFOS = INFO_SUBSET,
                   ecg_ts = norm_recds,
                   time_ts = time_ts,
                   group_var = 'rhy_grp',
                   group_colors = {'sinus': 'darkblue',
                                   abnormal_rhy_grp: 'red'},
                   group_labels = {'sinus': 'sinus, ' + str(age),
                                   abnormal_rhy_grp: abnormal_rhy_grp + ', ' + str(age + 1)},
                   x_label = x_label,
                   y_label = y_label,
                   x_lims = None,
                   show_plot = True)

# Remarks: af seems to have more negative Q and S depolarization in the QRS complex
# Remarks: af has two R-peaks which are higher than the other ones in the QRS complex
# Remarks: af QRS complex period is initially delayed wrt to the sinus one, but it progressivelly reduces its period and finishes in advance

abnormal_rhy_grp = 'af'
age = 59

INFO_SUBSET = ECG_COL_MAP[(ECG_COL_MAP.id.isin([130, 38])) & (ECG_COL_MAP.lead == 'i')]

plot_ecg_ts_by_grp(ECG_INFOS = INFO_SUBSET,
                   ecg_ts = norm_recds,
                   time_ts = time_ts,
                   group_var = 'rhy_grp',
                   group_colors = {'sinus': 'darkblue',
                                   abnormal_rhy_grp: 'red'},
                   group_labels = {'sinus': 'sinus, ' + str(age),
                                   abnormal_rhy_grp: abnormal_rhy_grp + ', ' + str(age)},
                   x_label = x_label,
                   y_label = y_label,
                   x_lims = None,
                   show_plot = True)

# Remarks: in between two QRS complexes of the sinus ecg, the number of af QRS patterns vary between 1 and 2

################################################################################
######## PLOTS 58-years-old SINUS ECG VS 58-years-old Tachycardia  #############
################################################################################

abnormal_rhy_grp = 'tachy'
age = 58

INFO_SUBSET = ECG_COL_MAP[(ECG_COL_MAP.id.isin([61, 114])) & (ECG_COL_MAP.lead == 'i')]

plot_ecg_ts_by_grp(ECG_INFOS = INFO_SUBSET,
                   ecg_ts = norm_recds,
                   time_ts = time_ts,
                   group_var = 'rhy_grp',
                   group_colors = {'sinus': 'darkblue',
                                   abnormal_rhy_grp: 'red'},
                   group_labels = {'sinus': 'sinus, ' + str(age),
                                   abnormal_rhy_grp: abnormal_rhy_grp + ', ' + str(age)},
                   x_label = x_label,
                   y_label = y_label,
                   x_lims = None,
                   show_plot = True)

# Remarks:

################################################################################
######## PLOTS 23-years-old SINUS ECG VS 23-years-old Tachycardia  #############
################################################################################

abnormal_rhy_grp = 'tachy'
age = 23

INFO_SUBSET = ECG_COL_MAP[(ECG_COL_MAP.id.isin([154, 117])) & (ECG_COL_MAP.lead == 'i')]

plot_ecg_ts_by_grp(ECG_INFOS = INFO_SUBSET,
                   ecg_ts = norm_recds,
                   time_ts = time_ts,
                   group_var = 'rhy_grp',
                   group_colors = {'sinus': 'darkblue',
                                   abnormal_rhy_grp: 'red'},
                   group_labels = {'sinus': 'sinus, ' + str(age),
                                   abnormal_rhy_grp: abnormal_rhy_grp + ', ' + str(age)},
                   x_label = x_label,
                   y_label = y_label,
                   x_lims = None,
                   show_plot = True)

# Remarks:


###############################################################################
######## PLOTS 34-years-old SINUS ECG VS 34-years-old Arrhythmia  #############
###############################################################################

abnormal_rhy_grp = 'arrhy'
age = 34

INFO_SUBSET = ECG_COL_MAP[(ECG_COL_MAP.id.isin([197, 58])) & (ECG_COL_MAP.lead == 'i')]

plot_ecg_ts_by_grp(ECG_INFOS = INFO_SUBSET,
                   ecg_ts = norm_recds,
                   time_ts = time_ts,
                   group_var = 'rhy_grp',
                   group_colors = {'sinus': 'darkblue',
                                   abnormal_rhy_grp: 'red'},
                   group_labels = {'sinus': 'sinus, ' + str(age),
                                   abnormal_rhy_grp: abnormal_rhy_grp + ', ' + str(age)},
                   x_label = x_label,
                   y_label = y_label,
                   x_lims = None,
                   show_plot = True)

# Remarks: in the arrhythmia signal there is a drop at the end with respect to the 0-line
