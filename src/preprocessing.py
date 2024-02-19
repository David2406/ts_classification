import wfdb
import os
import numpy as np
import pandas as pd
from __init__ import volta_logger
from src.utils import grid_projection_1D

sinus_vals = ['Sinus rhythm',
              'Sinus rhythm\nWandering atrial pacemaker']
brady_vals = ['Sinus bradycardia',
              'Sinus bradycardia\nWandering atrial pacemaker']
tachy_vals = ['Sinus tachycardia']
arrhy_vals = ['Sinus arrhythmia',
              'Sinus arrhythmia\nWandering atrial pacemaker']
af_vals = ['Atrial fibrillation',
           'Atrial fibrillation\nAberrant conduction',
           'Atrial flutter, typical']
ir_vals = ['Irregular sinus rhythm']

rhy_mapping = {'sinus': sinus_vals,
               'brady': brady_vals,
               'tachy': tachy_vals,
               'arrhy': arrhy_vals,
               'af': af_vals,
               'ir': ir_vals}


def load_ludb_records(ecg_folder_path = '/home/david/Dev/volta/ludb_1.0.1/data/',
                      nb_patients = 200):
    records_dfs = []

    for i in range(nb_patients):
        RECD = wfdb.rdrecord(record_name = os.path.join(ecg_folder_path, str(i + 1))).to_dataframe()
        RECD['id'] = i + 1
        records_dfs.append(RECD)

    RECDS = pd.concat(records_dfs)
    RECDS['time'] = RECDS.index.total_seconds()
    RECDS['time_delta_64'] = RECDS.index

    return RECDS.reset_index(drop = True)


def load_patient_df(patient_file_path = '/home/david/Dev/volta/ludb_1.0.1/ludb.csv',
                    rhy_var = 'Rhythms',
                    rhy_mapping = rhy_mapping,
                    logger = volta_logger):

    patient_df = pd.read_csv(patient_file_path)
    patient_df.rename(columns = {'ID': 'id'}, inplace = True)
    patient_df.loc[patient_df.Age == '>89\n', 'Age'] = '89\n'
    patient_df['Age'] = patient_df['Age'].apply(lambda x: int(x))

    logger.info('Projecting Age on a 10-years step grid, new column is called gAge')
    patient_df['gAge'] = patient_df['Age'].apply(lambda x: grid_projection_1D(x,
                                                                              x_origin = 10,
                                                                              grid_step = 10))

    logger.info('Grouping the different rhythms pathologies into macro categories. Grouped rhythms column is called rhy_grp')
    for rhy_key, rhy_vals in rhy_mapping.items():
        patient_df.loc[patient_df[rhy_var].isin(rhy_vals), 'rhy_grp'] = rhy_key

    return patient_df


lead_names = ['i',
              'ii',
              'iii',
              'avr',
              'avl',
              'avf',
              'v1',
              'v2',
              'v3',
              'v4',
              'v5',
              'v6']


def pivot_ecg_records(RECDS,
                      PATIENTS,
                      lead_cols = lead_names,
                      id_vars = ['id'],
                      time_vars = ['time'],
                      save_outputs = False,
                      pivot_recds_file_path = '/home/david/Dev/volta/saved_data/P_RECDS.pkl',
                      ecg_col_map_file_path = '/home/david/Dev/volta/saved_data/ECG_COL_MAP.pkl',
                      logger = volta_logger):

    P_RECDS = pd.pivot_table(RECDS,
                             values = lead_cols,
                             columns = id_vars,
                             index = time_vars)

    if save_outputs:
        logger.info('Saving pivot RECDS in %s', pivot_recds_file_path)
        P_RECDS.to_pickle(pivot_recds_file_path)

    ecg_col_mapping = {}
    for i, ecg_col_info in enumerate(P_RECDS.columns):
        ecg_col_mapping[ecg_col_info] = i

    ECG_COL_MAP = pd.DataFrame.from_dict(ecg_col_mapping,
                                         orient = 'index',
                                         columns = ['col_id'])
    ECG_COL_MAP['lead'] = [key[0] for key in ECG_COL_MAP.index]
    ECG_COL_MAP['id'] = [key[1] for key in ECG_COL_MAP.index]
    ECG_COL_MAP = pd.merge(ECG_COL_MAP,
                           PATIENTS,
                           on = 'id',
                           how = 'left')
    if save_outputs:
        logger.info('Saving pivot RECDS in %s', ecg_col_map_file_path)
        ECG_COL_MAP.to_pickle(ecg_col_map_file_path)

    return P_RECDS, ECG_COL_MAP


def norm_ecgs(P_RECDS,
              save_outputs = False,
              norm_ecg_file_path = '/home/david/Dev/volta/saved_data/norm_ecg.npy',
              logger = volta_logger):

    p_recds = P_RECDS.to_numpy()
    ts_mean_recds = np.mean(p_recds, axis = 0)
    ts_std_recds = np.std(p_recds, axis = 0)

    logger.info('Performing a Z-normalization for each ecg records (patient, lead)')
    norm_recds = (p_recds - ts_mean_recds) / ts_std_recds

    if save_outputs:
        np.save(norm_ecg_file_path, norm_recds)

    return norm_recds, ts_mean_recds, ts_std_recds
