# ecg-ai-project
High-recall PVC screening system evaluated patient-wise on MIT-BIH ECG data
# Patient-wise PVC Detection from ECG using CNNs

This project implements a **patient-wise evaluated ECG PVC detection system**
using beat-level convolutional neural networks (CNNs) on the
MIT-BIH Arrhythmia Database.

The focus is on **honest clinical evaluation**, not inflated performance.

---

## Clinical Motivation

Premature Ventricular Contractions (PVCs) are common ventricular arrhythmias
that are clinically relevant for cardiac screening and risk stratification.

Automated PVC detection systems must generalize to **unseen patients**.
Beat-wise random splits often cause patient leakage and overestimate performance.

This project explicitly avoids that.

---

## Key Design Principles

- **Patient-wise (record-wise) train/test split**
- **Beat-level morphology learning**
- **High-recall screening orientation**
- **Transparent evaluation and failure analysis**

---

## Dataset

- MIT-BIH Arrhythmia Database
- MLII lead
- Expert-annotated R-peaks and beat labels
- Binary classification:
  - Normal (N)
  - PVC (V)

---

## Pipeline Overview

ECG Signal  
→ High-pass filtering  
→ Beat extraction (0.2 s before, 0.4 s after R-peak)  
→ Beat normalization  
→ CNN probability prediction  
→ Evaluation on unseen patients  

---

## Model

- 1D CNN
- Two convolutional blocks
- ~100k parameters
- Output: PVC probability per beat

The model is intentionally simple to emphasize evaluation integrity.

---

## Evaluation Strategy

- **Record-wise split (patient-wise)**
- Test patients are never seen during training
- Metrics reported:
  - Precision
  - Recall
  - F1-score
  - Confusion matrix

Accuracy alone is not used as the primary metric.

---

## Results (Patient-wise)

### Without temporal post-processing

- PVC Recall: **0.93**
- PVC Precision: **0.38**
- Overall Accuracy: **0.75**

This configuration achieves **high sensitivity**, suitable for screening,
but produces many false positives.

### With naive temporal smoothing (2-of-3 rule)

- PVC Recall: **0.72**
- PVC Precision: **0.31**

Temporal smoothing reduced recall and did not improve precision,
indicating that naive beat-adjacency rules are insufficient
for PVC suppression in patient-wise settings.

---

## Key Insight

> Naive temporal smoothing can degrade PVC detection performance when
true PVCs are sparse or isolated. Rhythm-aware context is required
for effective alarm suppression.

This result highlights the importance of **system-level evaluation**
in healthcare AI.

---

## Limitations

- Beat-level analysis only
- No RR-interval or rhythm modeling
- MIT-BIH dataset bias
- Not intended for diagnostic use

---

## Intended Use

- High-recall PVC screening
- Holter-style review assistance
- Research and educational purposes

---

## How to Run

```bash
pip install -r requirements.txt
python train_recordwise.py
python evaluate.py
