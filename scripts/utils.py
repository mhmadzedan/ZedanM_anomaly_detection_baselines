import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score



def get_windows(X, window_size=40):
    Xw = []
    for i in range(len(X)-window_size+1):
        Xw.append(X[i:i+window_size])
    return np.array(Xw)


def transform_anomaly_scores(X_attack, scores):
    y_pred = np.zeros(X_attack.shape[0])
    counts = np.zeros(X_attack.shape[0])
    w_size = scores.shape[1]
    for i in range(len(scores)):
        i_score = scores[i]
        y_pred[i:i+w_size] += i_score
        counts[i:i+w_size] += 1
    return y_pred / counts


def calc_quality_metrics(y_true, y_pred, y_scores):
    """
    Calculate quality metrics for anomaly detection.
    Args:
        y_true: numpy.array, shape = (n_samples, )
            True labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.
        y_pred: numpy.array, shape = (n_samples, )
            Predicted labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.
        y_scores: numpy.array, shape = (n_samples, )
            Predicted anomaly scores.

    Returns:
        metrics = [ROC AUC, Recall, Precision, F1]

    Example:
        calc_quality_metrics(y_true=[0, 0, 1, 1, 1],
                             y_pred=[0, 1, 1, 1, 1],
                             y_scores=[-0.1, 0.2, 1, 10, 12])
    """
    rocauc = roc_auc_score(y_true, y_scores)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return [rocauc, recall, precision, f1]


def point_adjustment(y_true, y_pred):
    """
    Apply point adjustment for predicted labels as described in https://arxiv.org/pdf/1802.03903 (Fig. 7).
    Args:
        y_true: numpy.array, shape = (n_samples, )
            True labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.
        y_pred: numpy.array, shape = (n_samples, )
            Predicted labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.

    Returns:
        y_pred_pa: numpy.array, shape = (n_samples, )
            Adjusted predicted labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.
    """
    if len(y_true) != len(y_pred):
        raise Exception("y_true and y_pred must have the same length.")
    y_pred_pa = np.copy(y_pred)

    # find all segmetns of 1
    seg_start = None
    seg_end = None
    segment_inds = []
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if seg_start is None:
                seg_start = i
        elif y_true[i] == 0:
            if seg_start is not None:
                seg_end = i
                segment_inds.append([seg_start, seg_end])
            seg_start = None
            seg_end = None

    # adjust predictions
    for aseg in segment_inds:
        if np.sum(y_pred[aseg[0]:aseg[1]]) > 0:
            y_pred_pa[aseg[0]:aseg[1]] = 1

    return y_pred_pa


def best_quality_metrics(y_true, y_scores, use_point_adjustment=True,
                         min_anomaly_rate=0.001, max_anomaly_rate=1.0, step=0.01):
    """
    Select the best threshold for the quality metrics.
    Args:
        y_true: numpy.array, shape = (n_samples, )
            True labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.
        y_scores: numpy.array, shape = (n_samples, )
            Predicted anomaly scores.
        use_point_adjustment: boolean
            Apply point adjustment to the predicted labels or not. {True, False}
        min_anomaly_rate: float, [0, 1]
            The minimal anomaly rate in the prediction.
        max_anomaly_rate: float, [0, 1]
            The maximal anomaly rate in the prediction.
        step: float, [0, 1]
            Increasing step for the anomaly rate.

    Returns:
        best_metrics: [ROC AUC, Recall, Precision, F1]
            The best values of the qualuity metrics.
        best_thresh: float
            The best threshold for the predicted anomaly scores.

    """
    qs = np.arange(1 - max_anomaly_rate, 1 - min_anomaly_rate, step)
    best_f1 = -1
    best_metrics = None
    best_thresh = None
    for aq in qs:
        thresh = np.quantile(y_scores, aq)
        y_pred = 1 * (y_scores > thresh)
        if use_point_adjustment:
            y_pred = point_adjustment(y_true, y_pred)
        metrics = calc_quality_metrics(y_true, y_pred, y_scores)
        if metrics[3] > best_f1:
            best_f1 = metrics[3]
            best_metrics = metrics
            best_thresh = thresh
    return best_metrics, best_thresh




### AnomalyBERT quality metrics

import json, os
import numpy as np
import argparse
import matplotlib.pyplot as plt



# Exponential weighted moving average
def ewma(series, weighting_factor=0.9):
    current_factor = 1 - weighting_factor
    _ewma = series.copy()
    for i in range(1, len(_ewma)):
        _ewma[i] = _ewma[i-1] * weighting_factor + _ewma[i] * current_factor
    return _ewma


# Get anomaly sequences.
def anomaly_sequence(label):
    anomaly_args = np.argwhere(label).flatten()  # Indices for abnormal points.
    
    # Terms between abnormal invervals
    terms = anomaly_args[1:] - anomaly_args[:-1]
    terms = terms > 1

    # Extract anomaly sequences.
    sequence_args = np.argwhere(terms).flatten() + 1
    sequence_length = list(sequence_args[1:] - sequence_args[:-1])
    sequence_args = list(sequence_args)

    sequence_args.insert(0, 0)
    if len(sequence_args) > 1:
        sequence_length.insert(0, sequence_args[1])
    sequence_length.append(len(anomaly_args) - sequence_args[-1])

    # Get anomaly sequence arguments.
    sequence_args = anomaly_args[sequence_args]
    anomaly_label_seq = np.transpose(np.array((sequence_args, sequence_args + np.array(sequence_length))))
    return anomaly_label_seq, sequence_length


# Interval-dependent point
def interval_dependent_point(sequences, lengths):
    n_intervals = len(sequences)
    n_steps = np.sum(lengths)
    return (n_steps / n_intervals) / lengths


def f1_score_custom(gt, pr, anomaly_rate=0.05, adjust=True, modify=False):
    # get anomaly intervals
    gt_aug = np.concatenate([np.zeros(1), gt, np.zeros(1)]).astype(np.int32)
    gt_diff = gt_aug[1:] - gt_aug[:-1]

    begin = np.where(gt_diff == 1)[0]
    end = np.where(gt_diff == -1)[0]

    intervals = np.stack([begin, end], axis=1)

    # quantile cut
    pa = pr.copy()
    q = np.quantile(pa, 1-anomaly_rate)
    pa = (pa > q).astype(np.int32)
    
    # Modified F1
    if modify:
        gt_seq_args, gt_seq_lens = anomaly_sequence(gt)  # gt anomaly sequence args
        ind_p = interval_dependent_point(gt_seq_args, gt_seq_lens)  # interval-dependent point
        
        # Compute TP and FN.
        TP = 0
        FN = 0
        for _seq, _len, _p in zip(gt_seq_args, gt_seq_lens, ind_p):
            n_tp = pa[_seq[0]:_seq[1]].sum()
            n_fn = _len - n_tp
            TP += n_tp * _p
            FN += n_fn * _p
            
        # Compute TN and FP.
        TN = ((1 - gt) * (1 - pa)).sum()
        FP = ((1 - gt) * pa).sum()

    else:
        # point adjustment
        if adjust:
            for s, e in intervals:
                interval = slice(s, e)
                if pa[interval].sum() > 0:
                    pa[interval] = 1

        # confusion matrix
        TP = (gt * pa).sum()
        TN = ((1 - gt) * (1 - pa)).sum()
        FP = ((1 - gt) * pa).sum()
        FN = (gt * (1 - pa)).sum()

        assert (TP + TN + FP + FN) == len(gt)

    # Compute p, r, f1.
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2*precision*recall/(precision+recall)

    return precision, recall, f1_score


def anomalybert_quality_metrics(y_true, y_scores, verbose=1):
    
    # Compute roc auc
    rocauc = roc_auc_score(y_true, y_scores)
    
    # Compute precisions, recalls, F1-scores of the result.
    # Standard metrics
    best_eval = (0, 0, 0)
    best_rate = 0
    for rate in np.arange(0.001, 0.301, 0.001):
        evaluation = f1_score_custom(y_true, y_scores, rate, False)
        if evaluation[2] > best_eval[2]:
            best_eval = evaluation
            best_rate = rate
    if verbose == 1:
        print('Best F1-score without point adjustment')
        print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f} | ROC AUC: {rocauc:.5f}\n')

    # Metrics after point adjustment
    best_eval = (0, 0, 0)
    best_rate = 0
    for rate in np.arange(0.001, 0.301, 0.001):
        evaluation = f1_score_custom(y_true, y_scores, rate, True)
        if evaluation[2] > best_eval[2]:
            best_eval = evaluation
            best_rate = rate
    if verbose == 1:
        print('Best F1-score with point adjustment')
        print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f} | ROC AUC: {rocauc:.5f}\n')
        
    return (best_eval[0], best_eval[1], best_eval[2], rocauc) # precision, recall, f1, roc auc
