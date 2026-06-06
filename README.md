🌸 Project 2: Data Classification Using AI

## DecodeLabs Industrial Training Kit | Batch 2026

---

## 📌 Project Overview

This project implements a **Supervised Learning** classification model using the **K-Nearest Neighbors (KNN)** algorithm. Instead of writing explicit rules, the machine learns patterns from historical data to make intelligent decisions.

**Goal:** Build a basic classification model using a small dataset.

---

## 🏗️ Architectural Paradigm: IPO Framework

| Phase | Component | Implementation |
|-------|-----------|----------------|
| **I**nput | Iris Dataset | 150 samples, 4 features, 3 classes |
| **P**rocess | KNN Algorithm | Train-test split, feature scaling, classification |
| **O**utput | Model Evaluation | Confusion Matrix, F1 Score, Accuracy |

---

## 📊 Raw Material: The Iris Benchmark

| Property | Value |
|----------|-------|
| **Total Samples** | 150 (Balanced) |
| **Classes** | 3 (Setosa, Versicolor, Virginica) |
| **Dimensions** | 4 features |
| **Samples per class** | 50 |

### Features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

## 🔧 The Gatekeeper Rule: Feature Scaling

**Problem:** Raw data has different scales (bias)

**Solution:** StandardScaler - transforms data to have **Mean = 0, Variance = 1**
Tuning the Engine: Choosing "K"
Using the Elbow Method to find optimal K:

Test K values from 1 to 30

Find K with lowest error rate

Trade-off: Small K = overfitting, Large K = underfitting

K Value	Effect
Small K	High variance, sensitive to noise
Optimal K	Balanced performance
Large K	High bias, oversmoothing
🔍 Output Validation & Diagnostic Tools
The Diagnostic Tool: Confusion Matrix
text
              Predicted
              Set  Vers  Virg
Actual Set    TN    FP    FP
Actual Vers   FN    TP    FP
Actual Virg   FN    FN    TP
Key Metrics:
Accuracy: Overall correctness

Precision: Exactness of positive predictions

Recall: Completeness of positive predictions

F1 Score: Harmonic mean of precision and recall

Note: In imbalanced data, accuracy can be misleading. We must look deeper at F1 Score!

📊 Strategic Trade-Offs

Metric	Focus	Trade-off
Precision	Minimize False Positives	May miss some positives
Recall	Minimize False Negatives	May have more false alarms
F1 Score	Balance both	Best for imbalanced data
🎯 Results Dashboard
https://fraud_detection_results.png








