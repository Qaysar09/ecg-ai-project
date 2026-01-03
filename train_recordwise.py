import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from data_loader import load_ecg_beats
from model import build_cnn

# -------- RECORD-WISE SPLIT --------
TRAIN_RECORDS = ['100', '105', '106', '108', '109', '119']
TEST_RECORDS = ['200', '201', '202']
# ----------------------------------


print("Loading training data...")
X_train, y_train = load_ecg_beats(TRAIN_RECORDS)

print("Loading test data...")
X_test, y_test = load_ecg_beats(TEST_RECORDS)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train PVC:", (y_train == 1).sum())
print("Test PVC:", (y_test == 1).sum())

# -------- CLASS WEIGHTS --------
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight = {0: weights[0], 1: weights[1]}

# -------- MODEL --------
model = build_cnn(X_train.shape[1:])
model.summary()

# -------- TRAIN --------
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    class_weight=class_weight,
    verbose=1
)

model.save_weights("cnn.weights.h5")
print("Model weights saved.")
