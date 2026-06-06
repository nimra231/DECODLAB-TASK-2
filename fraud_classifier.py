import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets        import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (confusion_matrix,
                                     classification_report,
                                     f1_score, accuracy_score,
                                     roc_curve, auc,
                                     precision_recall_curve)
from collections             import Counter

print("\n" + "="*62)
print("  DecodeLabs — Project 2 : Credit Card Fraud Detection (KNN)")
print("="*62)

np.random.seed(42)
X, y = make_classification(
    n_samples       = 2000,
    n_features      = 10,
    n_informative   = 7,
    n_redundant      = 2,
    n_clusters_per_class = 1,
    weights         = [0.93, 0.07],   # 93% legit, 7% fraud — realistic ratio
    flip_y          = 0.01,
    random_state    = 42
)

feature_names = [
    "Transaction Amount",
    "Time Since Last Txn",
    "Merchant Risk Score",
    "Distance From Home",
    "Txn Frequency (24h)",
    "Country Match",
    "Device Trust Score",
    "Card Age (months)",
    "Velocity Score",
    "Account Balance Flag"
]

labels      = ["Legitimate", "Fraud"]
fraud_count = Counter(y)

print(f"\n✅  Dataset generated  →  {len(y)} transactions")
print(f"   Legitimate : {fraud_count[0]}  ({fraud_count[0]/len(y)*100:.1f}%)")
print(f"   Fraud      : {fraud_count[1]}  ({fraud_count[1]/len(y)*100:.1f}%)")
print(f"   Features   : {len(feature_names)}")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n✅  Features scaled  (mean=0, variance=1)")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size    = 0.20,
    random_state = 42,
    shuffle      = True,
    stratify     = y          # ensures fraud cases in both splits
)
print(f"\n✅  Stratified split  →  Train: {len(X_train)}  |  Test: {len(X_test)}")
print(f"   Fraud in test set : {sum(y_test)} cases")

print("\n⏳  Running Elbow Method (K = 1 to 30)...")

error_rates = []
f1_scores   = []
k_range     = range(1, 31)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, pred))
    f1_scores.append(f1_score(y_test, pred, average='weighted'))

optimal_k   = k_range[np.argmin(error_rates)]
best_f1_k   = k_range[np.argmax(f1_scores)]
print(f"✅  Optimal K (lowest error) → K = {optimal_k}")
print(f"   Best F1 K               → K = {best_f1_k}")

model = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')
model.fit(X_train, y_train)
predictions  = model.predict(X_test)
proba        = model.predict_proba(X_test)[:, 1]   # fraud probability

acc   = accuracy_score(y_test, predictions)
f1    = f1_score(y_test, predictions, average='weighted')
f1_fraud = f1_score(y_test, predictions, pos_label=1)
cm    = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions,
                                target_names=labels)

fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc     = auc(fpr, tpr)
precision_c, recall_c, _ = precision_recall_curve(y_test, proba)

tn, fp, fn, tp = cm.ravel()

print("\n" + "─"*62)
print(f"  ACCURACY       : {acc*100:.2f}%")
print(f"  F1 (weighted)  : {f1:.4f}")
print(f"  F1 (fraud only): {f1_fraud:.4f}")
print(f"  ROC-AUC        : {roc_auc:.4f}")
print(f"  True Positives : {tp}  (Fraud correctly caught)")
print(f"  False Negatives: {fn}  (Fraud missed — dangerous!)")
print(f"  False Positives: {fp}  (Legit flagged as fraud)")
print("─"*62)
print("\n  FULL CLASSIFICATION REPORT\n")
print(report)

DARK    = "#0A0F1E"
NAVY    = "#0D1B2A"
BLUE    = "#1A3A5C"
CYAN    = "#00D4FF"
RED     = "#FF3B5C"
GREEN   = "#00E676"
AMBER   = "#FFB300"
LGREY   = "#0F1923"
MID     = "#132030"
WHITE   = "#E8F4FD"

plt.rcParams.update({
    "figure.facecolor"  : DARK,
    "axes.facecolor"    : LGREY,
    "axes.edgecolor"    : BLUE,
    "axes.labelcolor"   : WHITE,
    "xtick.color"       : WHITE,
    "ytick.color"       : WHITE,
    "text.color"        : WHITE,
    "font.family"       : "monospace",
    "axes.grid"         : True,
    "grid.color"        : "#1A2A3A",
    "grid.linewidth"    : 0.5,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.spines.left"  : True,
    "axes.spines.bottom": True,
})

fig = plt.figure(figsize=(22, 15), facecolor=DARK)

fig.text(0.5, 0.975,
         "DecodeLabs  ·  PROJECT 2  ·  CREDIT CARD FRAUD DETECTION  ·  KNN CLASSIFIER",
         ha="center", fontsize=13, fontweight="bold",
         color=CYAN, family="monospace")
fig.text(0.5, 0.956,
         f"Dataset: 2,000 transactions  ·  Features: 10  ·  Optimal K: {optimal_k}  "
         f"·  Accuracy: {acc*100:.2f}%  ·  ROC-AUC: {roc_auc:.4f}",
         ha="center", fontsize=10, color=WHITE,
         alpha=0.6, family="monospace")

gs = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.44, wspace=0.34,
                        left=0.06, right=0.97,
                        top=0.92, bottom=0.07)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(k_range, error_rates, color=CYAN,
         linewidth=2.2, marker="o",
         markerfacecolor=DARK, markeredgecolor=CYAN,
         markeredgewidth=1.5, markersize=5)
ax1.plot(optimal_k, error_rates[optimal_k - 1],
         "o", color=RED, markersize=14, zorder=5)
ax1.annotate(f" K = {optimal_k}\n OPTIMAL",
             xy=(optimal_k, error_rates[optimal_k - 1]),
             xytext=(optimal_k + 3, error_rates[optimal_k - 1] + 0.008),
             fontsize=9, color=RED, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
ax1.set_facecolor(LGREY)
ax1.set_title("ELBOW CURVE  —  Finding Optimal K",
              fontsize=10, fontweight="bold", color=CYAN, pad=10)
ax1.set_xlabel("K Value")
ax1.set_ylabel("Error Rate")

ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap([[tn, fp], [fn, tp]],
            annot=[[f"TN\n{tn}", f"FP\n{fp}"],
                   [f"FN\n{fn}", f"TP\n{tp}"]],
            fmt="", cmap="Blues",
            xticklabels=["Predicted Legit", "Predicted Fraud"],
            yticklabels=["Actual Legit", "Actual Fraud"],
            linewidths=2, linecolor=DARK,
            annot_kws={"size": 13, "weight": "bold", "color": WHITE},
            cbar=False, ax=ax2)
ax2.set_facecolor(LGREY)
ax2.set_title("CONFUSION MATRIX",
              fontsize=10, fontweight="bold", color=CYAN, pad=10)
ax2.tick_params(axis="x", rotation=15, labelsize=9)
ax2.tick_params(axis="y", rotation=0, labelsize=9)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(fpr, tpr, color=CYAN, linewidth=2.5,
         label=f"ROC-AUC = {roc_auc:.4f}")
ax3.plot([0, 1], [0, 1], color=RED, linewidth=1.2,
         linestyle="--", alpha=0.6, label="Random Classifier")
ax3.fill_between(fpr, tpr, alpha=0.08, color=CYAN)
ax3.set_facecolor(LGREY)
ax3.set_title("ROC CURVE  —  Fraud Detection Power",
              fontsize=10, fontweight="bold", color=CYAN, pad=10)
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend(fontsize=9, facecolor=DARK, edgecolor=BLUE,
           labelcolor=WHITE)

ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(recall_c, precision_c, color=AMBER,
         linewidth=2.5)
ax4.fill_between(recall_c, precision_c, alpha=0.08, color=AMBER)
ax4.set_facecolor(LGREY)
ax4.set_title("PRECISION-RECALL  —  Fraud Class",
              fontsize=10, fontweight="bold", color=CYAN, pad=10)
ax4.set_xlabel("Recall  (Fraud Caught)")
ax4.set_ylabel("Precision  (Alerts Correct)")
ax4.annotate("High precision =\nfewer false alarms",
             xy=(0.2, 0.85), fontsize=8, color=AMBER,
             style="italic")

ax5 = fig.add_subplot(gs[1, 1])
fraud_mask  = y == 1
legit_mask  = y == 0
importance  = np.abs(X[fraud_mask].mean(axis=0) -
                     X[legit_mask].mean(axis=0))
sorted_idx  = np.argsort(importance)
bar_colors  = [CYAN if i == sorted_idx[-1] else BLUE
               for i in range(len(importance))]
bars = ax5.barh([feature_names[i] for i in sorted_idx],
                importance[sorted_idx],
                color=[bar_colors[i] for i in range(len(sorted_idx))],
                edgecolor=DARK, linewidth=0.8, height=0.6)
ax5.set_facecolor(LGREY)
ax5.set_title("FEATURE SEPARABILITY  —  Fraud vs Legit",
              fontsize=10, fontweight="bold", color=CYAN, pad=10)
ax5.set_xlabel("Mean Difference (Fraud − Legit)")
top_feat = feature_names[sorted_idx[-1]]
ax5.annotate(f"← Top signal",
             xy=(importance[sorted_idx[-1]], len(sorted_idx) - 1),
             xytext=(importance[sorted_idx[-1]] * 0.6, len(sorted_idx) - 1.5),
             fontsize=8, color=CYAN,
             arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.2))

ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(MID)
ax6.axis("off")

summary = [
    ("ALGORITHM",       "K-Nearest Neighbors"),
    ("DATASET",         "Credit Card Transactions"),
    ("TOTAL RECORDS",   "2,000  transactions"),
    ("FRAUD RATIO",     f"{fraud_count[1]/len(y)*100:.1f}%  of dataset"),
    ("FEATURES",        "10  financial signals"),
    ("SPLIT",           "80% train  /  20% test"),
    ("OPTIMAL K",       str(optimal_k)),
    ("ACCURACY",        f"{acc*100:.2f}%"),
    ("F1 (weighted)",   f"{f1:.4f}"),
    ("F1 (fraud)",      f"{f1_fraud:.4f}"),
    ("ROC-AUC",         f"{roc_auc:.4f}"),
    ("FRAUD CAUGHT",    f"{tp} / {tp+fn}  transactions"),
]

y_pos = 0.97
for key, val in summary:
    ax6.text(0.04, y_pos, key,
             fontsize=8.5, fontweight="bold",
             color=CYAN, transform=ax6.transAxes,
             family="monospace")
    ax6.text(0.48, y_pos, val,
             fontsize=8.5, color=WHITE,
             transform=ax6.transAxes,
             family="monospace")
    ax6.axhline(y=y_pos - 0.025,
                xmin=0.03, xmax=0.97,
                color="#1A2A3A", linewidth=0.6)
    y_pos -= 0.078

ax6.axvline(0.44, ymin=0.01, ymax=0.99,
            color=BLUE, linewidth=1.0)

for spine in ["top", "bottom", "left", "right"]:
    ax6.spines[spine].set_visible(True)
    ax6.spines[spine].set_color(CYAN)
    ax6.spines[spine].set_linewidth(1.5)

ax6.set_title("MODEL SUMMARY",
              fontsize=10, fontweight="bold", color=CYAN, pad=10)

plt.savefig("fraud_detection_results.png", dpi=180,
            bbox_inches="tight", facecolor=DARK)
print("\n✅  Dashboard saved  →  fraud_detection_results.png")
plt.show(block=True)

print("\n" + "="*62)
print("  Project 2 Complete! Badge Unlocked.")
print(f"  Fraud Detection System — {tp} out of {tp+fn} fraud cases caught.")
print("="*62 + "\n")
