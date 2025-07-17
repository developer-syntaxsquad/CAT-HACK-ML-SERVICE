import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("dataset.csv")

def train_model():
    # --- 1. Load Data ---
    

    # --- 2. Feature Engineering ---
    data["Fuel_Used"].replace(0, np.nan, inplace=True)
    data["Efficiency"] = data["Load_Cycles"] / data["Fuel_Used"]
    data["Efficiency"].replace([np.inf, -np.inf], np.nan, inplace=True)
    data["Efficiency"].fillna(data["Efficiency"].mean(), inplace=True)

    # --- 3. Define Features and Target ---
    X = data.drop(columns=["Prev_Task_Completion_Time", "Efficiency"])
    y = data["Efficiency"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())

    # --- 4. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 5. Preprocessing Pipelines ---
    cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_pipeline = Pipeline([("scaler", StandardScaler())])

    preprocessor = ColumnTransformer([
        ("cat", cat_pipeline, categorical_cols),
        ("num", num_pipeline, numerical_cols)
    ])

    # --- 6. Model Definitions ---
    regressors = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "XGBoost (SGD)": SGDRegressor(random_state=42, max_iter=1000, tol=1e-3),
        "SVR": SVR(),
        "Lasso": Lasso(alpha=0.1),
        "Ridge": Ridge(alpha=1.0),
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }

    # --- 7. Train and Evaluate Models ---
    results = []

    for name, model in regressors.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("regressor", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE (%)": mape
        })

    # --- 8. Results Summary ---
    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)

    print("\n🔍 Efficiency Prediction Results:")
    print(results_df)

    # --- 9. Train Best Model on Full Data ---
    best_model = GradientBoostingRegressor(random_state=42)
    pipeline_best = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", best_model)
    ])

    pipeline_best.fit(X_train, y_train)
    data["Predicted_Efficiency"] = pipeline_best.predict(X)

    # --- 10. Efficiency Rankings ---
    machine_efficiency = data.groupby("Machine_ID")["Predicted_Efficiency"].mean().sort_values(ascending=False)
    operator_efficiency = data.groupby("Operator_ID")["Predicted_Efficiency"].mean().sort_values(ascending=False)

    print("\n💡 Top 5 Machines by Avg Predicted Efficiency:")
    print(machine_efficiency.head())

    print("\n💡 Top 5 Operators by Avg Predicted Efficiency:")
    print(operator_efficiency.head())

    # --- 11. Predict Task Completion Time ---
    # Add predicted completion time (using some logic, assuming it's inversely proportional to efficiency)
    # You can replace this with a real model if available
    data["Predicted_Completion_Hours"] = data["Load_Cycles"] / data["Predicted_Efficiency"]

train_model()

# --- 12. Function to Predict Average Task Completion Time ---
def predict_average_time(machine_id, operator_id):
    machine_id = machine_id
    operator_id = operator_id

    filtered = data[(data["Machine_ID"] == machine_id) & (data["Operator_ID"] == operator_id)]

    if filtered.empty:
        print("❌ No matching records found for the given Machine_ID and Operator_ID.")
    else:
        avg_time = filtered["Predicted_Completion_Hours"].mean()
        return avg_time

