import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/data.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Target"])
y = df["Target"]


# ============================================================
# 3. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget encoding:")

for number, label in enumerate(label_encoder.classes_):
    print(f"{number} -> {label}")


# ============================================================
# 4. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. CREATE TUNED XGBOOST MODEL
# ============================================================

print("\nCreating tuned XGBoost model...")

model = XGBClassifier(

    objective="multi:softprob",

    num_class=3,

    # More trees, smaller learning rate
    n_estimators=500,

    learning_rate=0.03,

    # Control tree complexity
    max_depth=4,

    min_child_weight=2,

    # Randomization
    subsample=0.85,

    colsample_bytree=0.85,

    # Regularization
    reg_alpha=0.1,

    reg_lambda=1.5,

    gamma=0.1,

    random_state=42,

    eval_metric="mlogloss"
)


# ============================================================
# 6. TRAIN
# ============================================================

print("\n======================================")
print("TRAINING TUNED XGBOOST")
print("======================================")

model.fit(
    X_train,
    y_train
)

print("\nTraining completed!")


# ============================================================
# 7. PREDICT
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)


# ============================================================
# 8. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)


print("\n======================================")
print("TUNED MODEL PERFORMANCE")
print("======================================")

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"\nMacro F1: {macro_f1:.4f}"
)

print("\nClassification Report:")
print(report)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 11. CREATE DIRECTORIES
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 12. SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",

    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Tuned XGBoost - Student Outcome Prediction"
)

plt.tight_layout()

plt.savefig(
    "results/tuned_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/tuned_student_success_xgboost.pkl"
)

joblib.dump(
    label_encoder,
    "models/tuned_label_encoder.pkl"
)

joblib.dump(
    list(X.columns),
    "models/tuned_feature_names.pkl"
)


# ============================================================
# 14. SAVE REPORT
# ============================================================

with open(
    "results/tuned_classification_report.txt",
    "w"
) as file:

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Macro F1: {macro_f1:.4f}\n\n"
    )

    file.write(
        report
    )


# ============================================================
# 15. FINISHED
# ============================================================

print("\n======================================")
print("TUNED MODEL SAVED")
print("======================================")

print(
    "models/tuned_student_success_xgboost.pkl"
)

print(
    "models/tuned_label_encoder.pkl"
)

print(
    "models/tuned_feature_names.pkl"
)

print(
    "results/tuned_confusion_matrix.png"
)

print(
    "results/tuned_classification_report.txt"
)

print("\nTraining finished successfully!")