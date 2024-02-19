import numpy as np
import pandas as pd
import torch
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
