import wfdb
import numpy as np
from scipy.signal import butter, filtfilt

# -------------------------------------------------
# Configuration (frozen for the project)
# -------------------------------------------------
CLASSES = {'N': 0, 'V': 1}
WINDOW_BEFORE_SEC = 0.2
WINDOW_AFTER_SEC  = 0.4
CUTOFF_FREQ = 0.5


# -------------------------------------------------
# Signal preprocessing
# -------------------------------------------------
def highpass_filter(signal, fs):
    b, a = butter(2, CUTOFF_FREQ / (fs / 2), btype='high')
    return filtfilt(b, a, signal)


# -------------------------------------------------
# Main data loader
# -------------------------------------------------
def load_ecg_beats(records):
    """
    Load ECG beats from given MIT-BIH records.

    Parameters
    ----------
    records : list of str
        Record IDs (patients)

    Returns
    -------
    X : np.ndarray
        ECG beats, shape (N, samples, 1)
    y : np.ndarray
        Labels (0 = Normal, 1 = PVC)
    """

    X, y = [], []

    for rec in records:
        record = wfdb.rdrecord(rec, pn_dir="mitdb")
        ann = wfdb.rdann(rec, 'atr', pn_dir="mitdb")

        signal = record.p_signal[:, 0]
        fs = record.fs
        clean = highpass_filter(signal, fs)

        w_before = int(WINDOW_BEFORE_SEC * fs)
        w_after  = int(WINDOW_AFTER_SEC  * fs)

        for r, s in zip(ann.sample, ann.symbol):
            if s not in CLASSES:
                continue

            if r - w_before >= 0 and r + w_after < len(clean):
                beat = clean[r - w_before : r + w_after]

                # per-beat normalization
                beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

                X.append(beat)
                y.append(CLASSES[s])

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)

    return X, y
