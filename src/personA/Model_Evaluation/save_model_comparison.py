import pandas as pd

results = pd.DataFrame({
    "Model": [
        "Naive Bayes + SMOTENC",
        "Logistic Regression + SMOTENC"
    ],
    "Accuracy": [
        0.7308,
        0.6154
    ],
    "Class_1_Recall": [
        0.56,
        0.56
    ],
    "Class_1_F1": [
        0.59,
        0.50
    ],
    "ROC_AUC": [
        0.5948,
        0.5752
    ]
})

results.to_csv(
    "results/personA_model_comparison.csv",
    index=False
)

print("Model comparison saved!")
print()
print(results)