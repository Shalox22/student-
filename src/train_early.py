import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
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
print("Original shape:", df.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

TARGET = "Target"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 3. REMOVE FUTURE ACADEMIC PERFORMANCE FEATURES
# ============================================================

features_to_remove = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",

    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)"
]

print("\nRemoving future academic features:")

for feature in features_to_remove:
    print("-", feature)

X = X.drop(
    columns=features_to_remove
)


# ============================================================
# 4. SHOW FINAL FEATURES
# ============================================================

print("\n======================================")
print("EARLY-PREDICTION FEATURES")
print("======================================")

print("Number of features:", len(X.columns))

for i, column in enumerate(X.columns, start=1):
    print(f"{i}. {column}")


# ============================================================
# 5. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget encoding:")

for number, label in enumerate(label_encoder.classes_):
    print(f"{number} -> {label}")


# ============================================================
# 6. SPLIT DATA
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
# 7. CREATE XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,

    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    eval_metric="mlogloss"
)


# ============================================================
# 8. TRAIN
# ============================================================

print("\n======================================")
print("TRAINING EARLY-WARNING MODEL")
print("======================================")

model.fit(
    X_train,
    y_train
)

print("\nTraining completed!")


# ============================================================
# 9. PREDICT
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 10. EVALUATE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n======================================")
print("EARLY-WARNING MODEL PERFORMANCE")
print("======================================")

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(report)

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
    "Early Student Outcome Prediction"
)

plt.tight_layout()

plt.savefig(
    "results/early_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/early_student_success_xgboost.pkl"
)

joblib.dump(
    label_encoder,
    "models/early_label_encoder.pkl"
)

joblib.dump(
    list(X.columns),
    "models/early_feature_names.pkl"
)


# ============================================================
# 14. SAVE REPORT
# ============================================================

with open(
    "results/early_classification_report.txt",
    "w"
) as file:

    file.write(
        f"Accuracy: {accuracy:.4f}\n\n"
    )

    file.write(report)


# ============================================================
# 15. FINISHED
# ============================================================

print("\n======================================")
print("EARLY MODEL SAVED")
print("======================================")

print(
    "models/early_student_success_xgboost.pkl"
)

print(
    "models/early_label_encoder.pkl"
)

print(
    "models/early_feature_names.pkl"
)

print(
    "results/early_confusion_matrix.png"
)

print(
    "results/early_classification_report.txt"
)

print("\nDONE!")