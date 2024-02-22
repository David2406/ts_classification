import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.utils import grid_projection_1D
from scipy.fft import fft, fftfreq
from __init__ import volta_logger


def comp_fourier_spectrums(recds,
                           nb_sample_pts = 5000,
                           time_step = 1. / 500.,
                           freq_bounds = None,
                           save_outputs = False,
                           fourier_freqs_file_path = './ecg_freqs.npy',
                           fourier_ks_file_path = './ecg_ks.npy',
                           fourier_transform_file_path = './fft_recds.npy',
                           fourier_modes_ampl_file_path = './fft_modes_ampl.npy',
                           logger = volta_logger):

    logger.info('Computing Fourier frequencies based on %d sample points and a time step of %f (s)',
                nb_sample_pts, time_step)
    ecg_freqs = fftfreq(nb_sample_pts, time_step)[:nb_sample_pts // 2]

    logger.info('Turning the Fourier frequencies into a corresponding number of sample points')
    if freq_bounds is None:
        freq_low = 2. / nb_sample_pts
        freq_max = 0.5

    ecg_ks = np.round(1. / np.clip(ecg_freqs * time_step, freq_low, freq_max), 0).astype(int)

    logger.info('Computing Fourier transform on the time series')
    fft_recds = np.apply_along_axis(fft, axis = 0, arr = recds)

    logger.info('Computing Fourier modes amplitude for each frequency')
    fft_modes_ampl = np.apply_along_axis(lambda fft: (2 / nb_sample_pts) * np.abs(fft[0: nb_sample_pts // 2]),
                                         axis = 0,
                                         arr = fft_recds)

    if save_outputs:
        np.save(fourier_freqs_file_path, ecg_freqs)
        np.save(fourier_ks_file_path, ecg_ks)
        np.save(fourier_transform_file_path, fft_recds)
        np.save(fourier_modes_ampl_file_path, fft_modes_ampl)

    return ecg_freqs, ecg_ks, fft_recds, fft_modes_ampl


def get_cutting_frequency(lead_col_ids,
                          fft_modes_ampl,
                          ecg_freqs,
                          ecg_time_step,
                          ampl_factor = 0.05):

    lead_fft_modes_ampl = fft_modes_ampl[:, lead_col_ids]
    max_fft_ampl = np.max(lead_fft_modes_ampl)
    ampl_thres = ampl_factor * max_fft_ampl
    above_modes_indices = np.transpose((lead_fft_modes_ampl > ampl_thres).nonzero())
    cut_freq_index = np.max(above_modes_indices[:, 0])
    cut_freq = ecg_freqs[cut_freq_index]
    smooth_nb_pts = int(1 / (cut_freq * ecg_time_step))

    return max_fft_ampl, ampl_thres, cut_freq_index, cut_freq, smooth_nb_pts


def lead_cut_frequencies(ECG_COL_MAP,
                         time_step,
                         ecg_freqs,
                         fft_modes_ampl,
                         ampl_factor = 0.05,
                         cut_freqs_col_names = ['lead',
                                                'max_fft_ampl',
                                                'ampl_thres',
                                                'cut_freq_index',
                                                'cut_freq',
                                                'smooth_nb_pts'],
                         save_outputs = False,
                         cut_freqs_file_path = './CUT_FREQS.pkl',
                         logger = volta_logger):

    fourier_cut_infos = []

    for lead, ecg_info_df in ECG_COL_MAP.groupby(['lead']):

        logger.info('Estimating the cut frequency for lead %s, modes amplitude factor is set to %d %%',
                    lead, int(100 * ampl_factor))

        lead_col_ids = ecg_info_df['col_id'].to_numpy()
        cut_infos = get_cutting_frequency(lead_col_ids = lead_col_ids,
                                          fft_modes_ampl = fft_modes_ampl,
                                          ecg_freqs = ecg_freqs,
                                          ecg_time_step = time_step,
                                          ampl_factor = ampl_factor)

        fourier_cut_infos.append([lead] + list(cut_infos))

    CUT_FREQS = pd.DataFrame(data = fourier_cut_infos,
                             columns = cut_freqs_col_names)

    if save_outputs:
        CUT_FREQS.to_pickle(cut_freqs_file_path)

    return CUT_FREQS


def filter_wt_frequency(ECG_COL_MAP,
                        CUT_FREQS,
                        recds,
                        nb_sample_pts = 5000,
                        ks_var = 'smooth_nb_pts',
                        save_outputs = False,
                        filt_recds_file_path = './filtered_records.pt'):

    filtered_records = []

    for lead, ecg_info_df in ECG_COL_MAP.groupby(['lead']):

        lead_col_ids = ecg_info_df['col_id'].to_numpy()
        lead_recds = recds[:, lead_col_ids]
        t_lead_recds = torch.from_numpy(lead_recds)
        ks = int(CUT_FREQS[CUT_FREQS.lead == lead][ks_var])
        p_ks = int((ks - 1) / 2)
        lead_avg_pool = torch.nn.AvgPool1d(ks, stride = 1, padding = p_ks)
        lead_avg_recds = lead_avg_pool(torch.transpose(t_lead_recds, 0, 1))

        if lead_avg_recds.shape[1] < nb_sample_pts:
            lead_avg_recds = torch.column_stack((lead_avg_recds, lead_avg_recds[:, -1]))

        filt_recds = torch.transpose(lead_avg_recds, 0, 1)
        filtered_records.append(filt_recds)

    filtered_records = torch.cat(filtered_records, dim = 1)

    if save_outputs:
        torch.save(filtered_records, filt_recds_file_path)

    return filtered_records


def build_fourier_df(ecg_freqs,
                     fft_modes_ampl,
                     lead_names,
                     PATIENTS,
                     nb_leads = 12,
                     freq_grid_step = 10.,
                     save_outputs = False,
                     fourier_info_file_path = './FOURIER_INFOS.pkl',
                     agg_fourier_file_path = './AGG_FOURIER.pkl',
                     delete_tables = True,
                     logger = volta_logger):

    nb_patients = len(PATIENTS)
    flat_modes_ampl = fft_modes_ampl.flatten(order = 'F')
    freqs = np.tile(ecg_freqs, nb_patients * nb_leads)
    leads = np.repeat(lead_names, repeats = len(ecg_freqs) * nb_patients)
    patient_ids = np.tile(np.repeat(np.arange(1, 201, dtype = int), repeats = len(ecg_freqs)), nb_leads)

    logger.info('Building the Fourier frequencies / mode amplitudes table')
    fourier_data = np.stack([patient_ids, leads, freqs, flat_modes_ampl], axis = 1)

    FOURIER = pd.DataFrame(data = fourier_data,
                           columns = ['id', 'lead', 'freq', 'mode_ampl']).astype({'id': int})

    logger.info('Grouping frequencies by block of %d Hz', int(freq_grid_step))
    FOURIER['gfreq'] = FOURIER['freq'].apply(lambda x: grid_projection_1D(x,
                                                                          x_origin = 0.,
                                                                          grid_step = freq_grid_step))
    FOURIER = FOURIER.astype({'gfreq': int})

    if save_outputs:
        FOURIER.to_pickle(fourier_info_file_path)

    logger.info('Building the aggregated Fourier table, group vars are %s', str(['id', 'lead', 'gfreq']))
    AGG_FOURIER = FOURIER.groupby(['id', 'lead', 'gfreq'],
                                  as_index = False)['mode_ampl'].mean()
    if delete_tables:
        logger.info('Removing FOURIER table from local variables to save memory')
        del FOURIER

    AGG_FOURIER = pd.merge(AGG_FOURIER, PATIENTS, on = ['id'], how = 'left')
    AGG_FOURIER = AGG_FOURIER.groupby(['lead', 'gfreq', 'rhy_grp'],
                                      as_index = False)['mode_ampl'].mean()

    if save_outputs:
        AGG_FOURIER.to_pickle(agg_fourier_file_path)

    if delete_tables:
        return AGG_FOURIER

    return FOURIER, AGG_FOURIER


def plot_fourier_freq_hist(LEAD_AGG_FOURIER,
                           lead_var = 'lead',
                           width = 0.15,
                           save_plot = False,
                           return_axis = False,
                           show_plot = True,
                           save_filename = "./fourier_hist_avr.jpg"):

    lead_name = LEAD_AGG_FOURIER.iloc[0][lead_var]
    LF = LEAD_AGG_FOURIER.pivot(index = ['gfreq'],
                                values = ['mode_ampl'],
                                columns = 'rhy_grp')

    LF.columns = [lead for _, lead in LF.columns]
    lf_info = LF.to_dict()
    gfreqs = np.array(LF.index)
    x = np.arange(len(gfreqs))

    fig, ax = plt.subplots()

    multiplier = 0.

    for rhy_grp, ampl_hist_info in lf_info.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, [ampl for ampl in ampl_hist_info.values()],
                       width,
                       label = rhy_grp)
        multiplier += 1

    ax.set_xlabel('Frequencies (Hz)')
    ax.set_ylabel('Mean of Fourier mode amplitude')
    ax.set_xticks(x + width, gfreqs)
    ax.xaxis.grid(True, which = 'major')
    ax.yaxis.grid(True, which = 'major')
    ax.set_title('Fourier freqencies histogram, lead : {}'.format(lead_name))
    ax.legend(loc = 'upper right', ncols = 3)

    if show_plot:
        plt.show()
    if save_plot:
        fig.savefig(save_filename)

    if return_axis:
        return fig, ax
    else:
        plt.close()
