import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
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
# 4. TRAIN / TEST SPLIT
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
# 5. CREATE RANDOM FOREST MODEL
# ============================================================

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\n======================================")
print("TRAINING RANDOM FOREST")
print("======================================")

model.fit(
    X_train,
    y_train
)

print("\nTraining completed!")


# ============================================================
# 7. PREDICTIONS
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)


# ============================================================
# 8. CALCULATE METRICS
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

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n======================================")
print("RANDOM FOREST PERFORMANCE")
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

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 10. CREATE DIRECTORIES
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
# 11. SAVE CONFUSION MATRIX
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
    "Random Forest - Student Outcome Prediction"
)

plt.tight_layout()

plt.savefig(
    "results/random_forest_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/student_success_random_forest.pkl"
)

joblib.dump(
    label_encoder,
    "models/random_forest_label_encoder.pkl"
)

joblib.dump(
    list(X.columns),
    "models/random_forest_feature_names.pkl"
)


# ============================================================
# 13. SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    "results/random_forest_classification_report.txt",
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
# 14. FINISHED
# ============================================================

print("\n======================================")
print("RANDOM FOREST MODEL SAVED")
print("======================================")

print(
    "models/student_success_random_forest.pkl"
)

print(
    "models/random_forest_label_encoder.pkl"
)

print(
    "models/random_forest_feature_names.pkl"
)

print(
    "results/random_forest_confusion_matrix.png"
)

print(
    "results/random_forest_classification_report.txt"
)

print("\nTraining finished successfully!")