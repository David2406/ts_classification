import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from __init__ import volta_logger


def format_binary_classification(classifier,
                                 X,
                                 y,
                                 label_names = ['other', 'sinus'],
                                 pred_names = ['p_other', 'p_sinus']):

    preds = classifier.predict_proba(X)
    labels = np.transpose(np.stack([1 - y, y]))
    preds_data = np.stack([labels, preds], axis = 1).reshape(-1, 4)

    return pd.DataFrame(data = preds_data,
                        columns = label_names + pred_names)


def roc_curves_info(data,
                    label_names = ['u', 'c', 'i', 'o'],
                    score_names = ['p_u', 'p_c', 'p_i', 'p_o'],
                    micro_average = False,
                    macro_average = False,
                    logger = volta_logger):

    logger.info('Computing ROC curve information related to labels %s', str(label_names))
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    nb_classes = len(label_names)

    for i in np.arange(nb_classes):
        fpr[label_names[i]], tpr[label_names[i]], thresholds[label_names[i]] = roc_curve(data[label_names[i]],
                                                                                         data[score_names[i]])
        roc_auc[label_names[i]] = auc(fpr[label_names[i]],
                                      tpr[label_names[i]])

    if micro_average:
        all_labels = np.array(data[label_names]).ravel()
        all_scores = np.array(data[score_names]).ravel()
        fpr['micro'], tpr['micro'], thresholds['micro'] = roc_curve(all_labels, all_scores)
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    if macro_average:
        all_fpr = np.unique(np.concatenate([fpr[label] for label in label_names]))
        mean_tpr = np.zeros_like(all_fpr)
        for label in label_names:
            mean_tpr += np.interp(all_fpr, fpr[label], tpr[label])

        mean_tpr /= nb_classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    return fpr, tpr, thresholds, roc_auc
