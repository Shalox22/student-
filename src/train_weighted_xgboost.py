import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
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
print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURES AND TARGET
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
# 5. DEFINE WEIGHT EXPERIMENTS
# ============================================================

# Class mapping:
#
# 0 = Dropout
# 1 = Enrolled
# 2 = Graduate

weight_experiments = {
    "baseline": {
        0: 1.0,
        1: 1.0,
        2: 1.0
    },

    "enrolled_1.25": {
        0: 1.0,
        1: 1.25,
        2: 1.0
    },

    "enrolled_1.5": {
        0: 1.0,
        1: 1.5,
        2: 1.0
    },

    "enrolled_1.75": {
        0: 1.0,
        1: 1.75,
        2: 1.0
    },

    "enrolled_2.0": {
        0: 1.0,
        1: 2.0,
        2: 1.0
    }
}


# ============================================================
# 6. STORE RESULTS
# ============================================================

results = []

best_model = None
best_name = None
best_macro_f1 = -1
best_accuracy = 0


# ============================================================
# 7. RUN EXPERIMENTS
# ============================================================

for experiment_name, class_weights in weight_experiments.items():

    print("\n")
    print("======================================")
    print(f"EXPERIMENT: {experiment_name}")
    print("======================================")

    print("Class weights:", class_weights)

    # Create sample weights for each training row
    sample_weights = np.array([
        class_weights[int(label)]
        for label in y_train
    ])

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

    print("Training...")

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    enrolled_f1 = report_dict["Enrolled"]["f1-score"]

    enrolled_recall = report_dict["Enrolled"]["recall"]

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    print(
        f"Enrolled F1: {enrolled_f1:.4f}"
    )

    print(
        f"Enrolled Recall: {enrolled_recall:.4f}"
    )

    results.append({
        "experiment": experiment_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "enrolled_f1": enrolled_f1,
        "enrolled_recall": enrolled_recall
    })

    # Select model primarily by macro F1
    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_accuracy = accuracy
        best_model = model
        best_name = experiment_name


# ============================================================
# 8. DISPLAY COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("======================================")
print("WEIGHTED XGBOOST COMPARISON")
print("======================================")

print(
    results_df.to_string(index=False)
)


# ============================================================
# 9. BEST MODEL
# ============================================================

print("\n")
print("======================================")
print("BEST WEIGHTED MODEL")
print("======================================")

print(
    f"Experiment: {best_name}"
)

print(
    f"Accuracy: {best_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: {best_macro_f1:.4f}"
)


# ============================================================
# 10. FINAL PREDICTIONS
# ============================================================

best_predictions = best_model.predict(
    X_test
)

best_report = classification_report(
    y_test,
    best_predictions,
    target_names=label_encoder.classes_
)

best_cm = confusion_matrix(
    y_test,
    best_predictions
)

print("\nFinal Classification Report:")
print(best_report)

print("\nFinal Confusion Matrix:")
print(best_cm)


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
# 12. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "results/weighted_xgboost_comparison.csv",
    index=False
)


# ============================================================
# 13. SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    best_cm,
    annot=True,
    fmt="d",

    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    f"Weighted XGBoost - {best_name}"
)

plt.tight_layout()

plt.savefig(
    "results/weighted_xgboost_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/student_success_weighted_xgboost.pkl"
)

joblib.dump(
    label_encoder,
    "models/weighted_xgboost_label_encoder.pkl"
)

joblib.dump(
    list(X.columns),
    "models/weighted_xgboost_feature_names.pkl"
)


# ============================================================
# 15. SAVE REPORT
# ============================================================

with open(
    "results/weighted_xgboost_report.txt",
    "w"
) as file:

    file.write(
        "Weighted XGBoost Model Comparison\n"
    )

    file.write(
        "=================================\n\n"
    )

    file.write(
        results_df.to_string(index=False)
    )

    file.write(
        "\n\nBest Model:\n"
    )

    file.write(
        f"Experiment: {best_name}\n"
    )

    file.write(
        f"Accuracy: {best_accuracy:.4f}\n"
    )

    file.write(
        f"Macro F1: {best_macro_f1:.4f}\n\n"
    )

    file.write(
        best_report
    )


# ============================================================
# 16. FINISHED
# ============================================================

print("\n")
print("======================================")
print("WEIGHTED XGBOOST FINISHED")
print("======================================")

print(
    "\nSaved comparison:"
)

print(
    "results/weighted_xgboost_comparison.csv"
)

print(
    "\nSaved model:"
)

print(
    "models/student_success_weighted_xgboost.pkl"
)

print(
    "\nTraining completed successfully!")