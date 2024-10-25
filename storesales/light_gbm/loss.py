import numpy as np


def clipped_rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))
