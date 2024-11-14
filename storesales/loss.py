import numpy as np


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred))))


def clipped_rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return rmsle(y_true, y_pred)
