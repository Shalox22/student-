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
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/data.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. SHOW TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(df["Target"].value_counts())


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Target"])
y = df["Target"]


# ============================================================
# 4. ENCODE TARGET LABELS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget encoding:")

for number, label in enumerate(label_encoder.classes_):
    print(f"{number} -> {label}")


# ============================================================
# 5. SPLIT DATA
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
# 6. CREATE XGBOOST MODEL
# ============================================================

print("\nCreating XGBoost model...")

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
# 7. TRAIN MODEL
# ============================================================

print("\n======================================")
print("STARTING MODEL TRAINING")
print("======================================")

model.fit(
    X_train,
    y_train
)

print("\nTraining completed successfully!")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)


# ============================================================
# 9. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)

print("\nClassification Report:")
print(report)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 12. CREATE OUTPUT DIRECTORIES
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
# 13. SAVE CONFUSION MATRIX
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
    "Student Academic Success Prediction"
)

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "models/student_success_xgboost.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

joblib.dump(
    list(X.columns),
    "models/feature_names.pkl"
)


# ============================================================
# 15. SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    "results/classification_report.txt",
    "w"
) as file:

    file.write(
        f"Accuracy: {accuracy:.4f}\n\n"
    )

    file.write(
        report
    )


# ============================================================
# 16. FINISHED
# ============================================================

print("\n======================================")
print("TRAINING FINISHED")
print("======================================")

print("\nGenerated files:")

print(
    "\nModel:"
)

print(
    "models/student_success_xgboost.pkl"
)

print(
    "\nLabel encoder:"
)

print(
    "models/label_encoder.pkl"
)

print(
    "\nFeature names:"
)

print(
    "models/feature_names.pkl"
)

print(
    "\nConfusion matrix:"
)

print(
    "results/confusion_matrix.png"
)

print(
    "\nClassification report:"
)

print(
    "results/classification_report.txt"
)

print(
    "\nYour AI model has been trained successfully!"
)