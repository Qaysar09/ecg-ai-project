import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import load_ecg_beats
from model import build_cnn

TEST_RECORDS = ['200', '201', '202']


def temporal_smoothing(y_prob, threshold=0.5, window=3, min_positive=2):
    y_smooth = []

    for i in range(len(y_prob)):
        start = max(0, i - window + 1)
        window_probs = y_prob[start:i+1]
        positives = np.sum(window_probs > threshold)

        y_smooth.append(1 if positives >= min_positive else 0)

    return np.array(y_smooth)


print("Loading test data...")
X_test, y_test = load_ecg_beats(TEST_RECORDS)

model = build_cnn(X_test.shape[1:])
model.load_weights("cnn.weights.h5")

y_prob = model.predict(X_test).ravel()

# ---- WITHOUT SMOOTHING ----
y_raw = (y_prob > 0.5).astype(int)

print("\nWITHOUT TEMPORAL SMOOTHING")
print(confusion_matrix(y_test, y_raw))
print(classification_report(y_test, y_raw))

# ---- WITH SMOOTHING ----
y_smooth = temporal_smoothing(y_prob)

print("\nWITH TEMPORAL SMOOTHING")
print(confusion_matrix(y_test, y_smooth))
print(classification_report(y_test, y_smooth))
