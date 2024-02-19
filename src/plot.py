import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


def plot_by_group(data,
                  group_var = 'welding_state',
                  group_colors = {'underdiluted': 'darkblue',
                                  'correct': 'darkblue',
                                  'overdiluted': 'darkred'},
                  group_labels = {'underdiluted': 'dilution < 10%',
                                  'correct': 'dilution [10, 60]%',
                                  'overdiluted': 'dilution > 60 %'},
                  x_var = 'smd',
                  y_var = 'ratio',
                  x_label = 'Surface métal déposé mm²',
                  y_label = 'Ratio épaisseur / largeur cordon',
                  legend_loc = 'lower right',
                  marker_size = 3,
                  plot_style = 'o',
                  fig_size = (10, 6),
                  xtick_rotation = 45,
                  x_lims = None,
                  y_lims = None,
                  save_plot = False,
                  return_axis = False,
                  show_plot = True,
                  save_filename = "./welding_quality_smd_ratio_200.jpg"):

    fig, ax = plt.subplots(figsize = fig_size)

    for key, grp in data.groupby(group_var):

        plt.plot(grp[x_var],
                 grp[y_var],
                 plot_style,
                 markersize = marker_size,
                 color = group_colors[key],
                 label = group_labels[key])

    ax.legend(loc = legend_loc)
    ax.xaxis.grid(True, which = 'major')
    ax.yaxis.grid(True, which = 'major')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.xticks(rotation = xtick_rotation)
    plt.yticks()

    if not x_lims is None:
        plt.xlim(x_lims[0], x_lims[1])
    if not y_lims is None:
        plt.ylim(y_lims[0], y_lims[1])

    if show_plot:
        plt.show()
    if save_plot:
        fig.savefig(save_filename)

    if return_axis:
        return fig, ax
    else:
        plt.close()


def plot_ecg_ts(ECG_INFOS,
                ecg_ts,
                time_ts,
                cmap,
                x_ticks,
                y_ticks,
                x_label,
                y_label,
                legend_loc = 'lower right',
                x_lims = None,
                x_rotation = 45,
                save_plot = False,
                return_axis = False,
                show_plot = True,
                save_filename = "./out.jpeg"):

    seen_ages = []
    fig, ax = plt.subplots()

    for patient_id, ecg_info_df in ECG_INFOS.groupby(['id']):
        grid_age = ecg_info_df.iloc[0]['gAge']
        col_id = ecg_info_df.iloc[0]['col_id']

        plt.plot(time_ts,
                 ecg_ts[:, col_id],
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

    plt.xticks(rotation = x_rotation)

    if show_plot:
        plt.show()
    if save_plot:
        fig.savefig(save_filename)

    if return_axis:
        return fig, ax
    else:
        plt.close()


def plot_ecg_ts_by_grp(ECG_INFOS,
                       ecg_ts,
                       time_ts,
                       group_var = 'rhy_grp',
                       group_colors = {'sinus': 'darkblue',
                                       'brady': 'red'},
                       group_labels = {'sinus': 'sinus, ',
                                       'brady': 'brady'},
                       x_label = 'x',
                       y_label = 'y',
                       legend_loc = 'lower right',
                       x_lims = None,
                       x_rotation = 45,
                       save_plot = False,
                       return_axis = False,
                       show_plot = True,
                       save_filename = "./out.jpeg"):

    fig, ax = plt.subplots()

    for patient_id, ecg_info_df in ECG_INFOS.groupby(['id']):
        col_id = ecg_info_df.iloc[0]['col_id']
        grp = ecg_info_df.iloc[0][group_var]
        color = group_colors[grp]
        label = group_labels[grp]

        plt.plot(time_ts,
                 ecg_ts[:, col_id],
                 '-',
                 color = color,
                 label = label)

    ax.legend(loc = legend_loc)
    ax.xaxis.grid(True, which = 'major')
    ax.yaxis.grid(True, which = 'major')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if not x_lims is None:
        plt.xlim(x_lims[0], x_lims[1])

    plt.xticks(rotation = x_rotation)

    if show_plot:
        plt.show()
    if save_plot:
        fig.savefig(save_filename)

    if return_axis:
        return fig, ax
    else:
        plt.close()


rhy_color_mapping = {'sinus': 'blue',
                     'brady': 'green',
                     'af': 'red',
                     'arrhy': 'orange',
                     'tachy': 'purple',
                     'ir': 'brown'}

rhy_label_mapping = {'sinus': 'sinus',
                     'brady': 'brady',
                     'af': 'af',
                     'arrhy': 'arrhy',
                     'tachy': 'tachy',
                     'ir': 'ir'}


def plot_fourier_ecg(ECG_INFOS,
                     freqs,
                     fft_recds,
                     rhy_color_mapping = {'sinus': 'blue',
                                          'brady': 'green',
                                          'af': 'red',
                                          'arrhy': 'orange',
                                          'tachy': 'purple',
                                          'ir': 'brown'},
                     rhy_label_mapping = {'sinus': 'sinus',
                                          'brady': 'brady',
                                          'af': 'af',
                                          'arrhy': 'arrhy',
                                          'tachy': 'tachy',
                                          'ir': 'ir'},
                     x_label = 'Fourier frequency (Hz)',
                     y_label = 'Fourier mode amplitude',
                     x_lims = None,
                     legend_loc = 'lower right',
                     save_plot = False,
                     return_axis = False,
                     show_plot = True,
                     save_filename = "./out.jpeg"):

    nb_pts = fft_recds.shape[0]
    fig, ax = plt.subplots()

    seen_labels = set()
    for patient_id, ecg_info_df in ECG_INFOS.groupby(['id']):
        col_id = ecg_info_df.iloc[0]['col_id']
        rhy_grp = ecg_info_df.iloc[0]['rhy_grp']
        color = rhy_color_mapping[rhy_grp]
        label = rhy_label_mapping[rhy_grp]

        fft_ecg = fft_recds[:, col_id]

        plt.plot(freqs,
                 (2.0 / nb_pts) * np.abs(fft_ecg[0: nb_pts // 2]),
                 '-',
                 color = color,
                 label = label if not (label in seen_labels) else None)
        seen_labels.add(label)

    ax.legend(loc = legend_loc)
    ax.xaxis.grid(True, which = 'major')
    ax.yaxis.grid(True, which = 'major')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if not x_lims is None:
        plt.xlim(x_lims[0], x_lims[1])

    if show_plot:
        plt.show()
    if save_plot:
        fig.savefig(save_filename)

    if return_axis:
        return fig, ax
    else:
        plt.close()


def plot_roc_curve(fpr,
                   tpr,
                   roc_auc,
                   label_names,
                   label_colors,
                   lw = 2,
                   x_label = 'False Positve Rate',
                   y_label = 'True Positive Rate',
                   legend_loc = 'upper right',
                   save_plot = False,
                   return_axis = False,
                   show_plot = True,
                   save_filename = "./out.jpeg"):

    fig, ax = plt.subplots()

    for label, color in zip(label_names, label_colors):
        ax.plot(fpr[label],
                tpr[label],
                color = color,
                label = "ROC curve of class {0} (auc = {1:0.2f})".format(label, roc_auc[label]))

        ax.plot([0, 1], [0, 1], "k--", lw = lw)
        ax.legend(loc = legend_loc)
        ax.grid()

    ax.legend(loc = legend_loc)
    ax.xaxis.grid(True, which = 'major')
    ax.yaxis.grid(True, which = 'major')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    if show_plot:
        plt.show()
    if save_plot:
        fig.savefig(save_filename)
    if return_axis:
        return fig, ax
    else:
        plt.close()
