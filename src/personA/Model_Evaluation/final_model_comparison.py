import pandas as pd

# Results from our experiments
results = pd.DataFrame({
    "Model": [
        "Majority-Class Baseline",
        "Naive Bayes",
        "Naive Bayes + SMOTENC",
        "Logistic Regression + SMOTENC"
    ],
    "Accuracy": [
        65.38,
        65.38,
        73.08,
        61.54
    ],
    "Class_1_Recall": [
        0.00,
        22.00,
        56.00,
        56.00
    ],
    "Class_1_F1": [
        0.00,
        31.00,
        59.00,
        50.00
    ],
    "ROC_AUC": [
        None,
        None,
        59.48,
        57.52
    ]
})

print("FINAL MODEL COMPARISON")
print("======================")

print(results.to_string(index=False))

# Find best model based on accuracy
best_accuracy = results.loc[
    results["Accuracy"].idxmax()
]

print("\nBest model based on accuracy:")
print(best_accuracy["Model"])

print("Accuracy:", best_accuracy["Accuracy"], "%")

# Find best model based on high-risk recall
best_recall = results.loc[
    results["Class_1_Recall"].idxmax()
]

print("\nBest model based on high-risk recall:")
print(best_recall["Model"])

print("Class 1 recall:", best_recall["Class_1_Recall"], "%")

# Save final comparison
results.to_csv(
    "results/personA_final_model_comparison.csv",
    index=False
)

print("\nSaved:")
print("results/personA_final_model_comparison.csv")