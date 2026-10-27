# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "155e4fde-1635-4d52-8138-75a51f15cff7",
# META       "default_lakehouse_name": "bdca",
# META       "default_lakehouse_workspace_id": "57a287bc-575d-4670-b3ec-ffd670392cc7",
# META       "known_lakehouses": [
# META         {
# META           "id": "155e4fde-1635-4d52-8138-75a51f15cff7"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# Welcome to your new notebook
# Type here in the cell editor to add code!


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

%pip install xgboost


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print("Reason for Churn experiment")
print("Notebook connected successfully")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
import numpy as np

file_path = "/lakehouse/default/Files/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path, header=0)

print("Dataset loaded successfully")
print("Number of rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

display(df.head())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

target = "Churn"

df.columns = df.columns.str.strip()

required_columns = features + [target]
missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("Missing columns:", missing_columns)
else:
    print("All seven input features and Churn are available.")
    print("\nChurn distribution:")
    print(df["Churn"].value_counts(dropna=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Select exactly the seven features and the target
features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

target = "Churn"

data = df[features + [target]].copy()

# Convert numerical columns into numeric format
for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Remove rows without a valid target
data = data[data["Churn"].isin(["Yes", "No"])].copy()

X = data[features]
y = data["Churn"].map({"No": 0, "Yes": 1})

# Define categorical and numerical features
categorical_features = [
    "gender", "Partner", "Dependents"
]

numerical_features = [
    "SeniorCitizen", "tenure",
    "MonthlyCharges", "TotalCharges"
]

# Build preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ),
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical_features
        )
    ]
)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Preprocessing completed successfully!")
print("Total records:", len(data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training churn rate:", round(y_train.mean() * 100, 2), "%")
print("Testing churn rate:", round(y_test.mean() * 100, 2), "%")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Reload the CSV because the session restarted
file_path = "/lakehouse/default/Files/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path, header=0)

# Select the seven features and target
features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

target = "Churn"

df.columns = df.columns.str.strip()
data = df[features + [target]].copy()

# Convert numeric columns
for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Keep rows with valid target values
data = data[data[target].isin(["Yes", "No"])].copy()

X = data[features]
y = data[target].map({"No": 0, "Yes": 1})

categorical_features = ["gender", "Partner", "Dependents"]
numerical_features = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
]

preprocessor = ColumnTransformer([
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features),

    ("numerical", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Step 5 completed successfully!")
print("Total records:", len(data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Features:", features)
print("Target: Churn")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

print("All four machine-learning algorithms imported successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
}

print("Four models defined successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

results = []
trained_models = {}

for model_name, model in models.items():

    print(f"Training {model_name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    results.append(metrics)
    trained_models[model_name] = pipeline

    print(f"{model_name} completed.")

print("\nAll four models have been trained successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

display(
    results_df.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
        "ROC-AUC": "{:.4f}"
    })
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

for model_name, pipeline in trained_models.items():

    y_pred = pipeline.predict(X_test)

    print(f"\nConfusion Matrix — {model_name}")
    print("Rows = Actual; Columns = Predicted")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

results_df.to_csv(
    "/tmp/reason_for_churn_model_results.csv",
    index=False
)

print("Model comparison results saved successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

results_df.to_csv(
    "/lakehouse/default/Files/reason_for_churn_model_results.csv",
    index=False
)

print("Results saved to the bdca Lakehouse Files folder.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import mlflow

mlflow.set_experiment("Reason for Churn")

for model_name, pipeline in trained_models.items():

    row = results_df.loc[
        results_df["Model"] == model_name
    ].iloc[0]

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", "Churn")
        mlflow.log_param(
            "features",
            ",".join(features)
        )
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]:
            mlflow.log_metric(
                metric.replace(" ", "_").replace("-", "_").lower(),
                float(row[metric])
            )

print("MLflow experiment logging completed.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import mlflow

# Use underscores instead of spaces
mlflow.set_experiment("Reason_for_Churn")

for model_name, pipeline in trained_models.items():

    row = results_df.loc[
        results_df["Model"] == model_name
    ].iloc[0]

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", "Churn")
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]:
            metric_name = (
                metric.replace(" ", "_")
                      .replace("-", "_")
                      .lower()
            )

            mlflow.log_metric(
                metric_name,
                float(row[metric])
            )

print("MLflow experiment logging completed successfully!")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(
    results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).round(4)
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

best_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

best_model_name = best_row["Model"]

print("Best model based on F1 Score:", best_model_name)
print("\nPerformance metrics:")

for metric in [
    "Accuracy", "Precision", "Recall",
    "F1 Score", "ROC-AUC"
]:
    print(f"{metric}: {best_row[metric]:.4f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from sklearn.inspection import permutation_importance

best_model = trained_models[best_model_name]

importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean,
    "Std_Dev": importance.importances_std
}).sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

display(importance_df.round(4))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

importance_df.to_csv(
    "/lakehouse/default/Files/churn_feature_importance.csv",
    index=False
)

print("Feature importance saved to Lakehouse Files.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd

# Select two customer records from your existing dataset
sample_customers = X_test.head(2).copy()

# Use the best model selected earlier
best_model = trained_models[best_model_name]

# Predict churn (0 = No, 1 = Yes)
predictions = best_model.predict(sample_customers)

# Predict churn probability
probabilities = best_model.predict_proba(sample_customers)[:, 1]

# Display the results
prediction_results = sample_customers.copy()

prediction_results["Actual_Churn"] = (
    y_test.loc[sample_customers.index]
    .map({0: "No", 1: "Yes"})
    .values
)

prediction_results["Predicted_Churn"] = pd.Series(
    predictions,
    index=sample_customers.index
).map({0: "No", 1: "Yes"}).values

prediction_results["Churn_Probability_%"] = (
    probabilities * 100
).round(2)

display(prediction_results)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd

# Enter two customer records
new_customers = pd.DataFrame([
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "MonthlyCharges": 85.50,
        "TotalCharges": 427.50
    },
    {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 48,
        "MonthlyCharges": 65.25,
        "TotalCharges": 3132.00
    }
])

# Use your trained model
model = trained_models[best_model_name]

# Predict churn and probability
predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)[:, 1]

# Display results
output = new_customers.copy()
output["Predicted_Churn"] = pd.Series(predictions).map(
    {0: "No", 1: "Yes"}
).values
output["Churn_Probability_%"] = (probabilities * 100).round(2)

print("Model used:", best_model_name)
display(output)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print("Original gender values:")
print(df["gender"].value_counts())

# Create a numeric gender column
df["gender_numeric"] = df["gender"].map({
    "Male": 1,
    "Female": 2
})

print("\nNumeric gender values:")
print(df[["gender", "gender_numeric"]].drop_duplicates())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
