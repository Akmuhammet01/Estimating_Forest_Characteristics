# command to install cuML library
# !pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import joblib
import os

# cuml libraries for GPU Acceleration
import cuml.accel
cuml.accel.install()
import cupy as cp
from cuml.ensemble import RandomForestRegressor as cuRFR

# Parameter to change
data_folder_path = "/content/drive/MyDrive/Thesis"
save_dir = "/content/drive/MyDrive/Thesis/"
grid_search = True
data = "AEF"  # S2 for sentinal-2 data

# File path preparation
file_name = f"full_shuffled_{data.lower()}_data.csv"
file_path = os.path.join(data_folder_path, file_name)

print("Loading data...")
df = pd.read_csv(file_path)
print("Data is loaded")

date_column_name = f"{data}_Date"

print("\nData preparation...")
df.drop(columns=["age", "stock_per_ha", "basal_area", "poly_id", date_column_name], axis=1, inplace=True)

y = df["height"]
X = df.drop(columns="height", axis=1)

# Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"\nDatasets are read! X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

if grid_search:
    print("\nGrid search training on GPU...")

    # XGBRegressor
    param_grid_xgb = {
        'n_estimators': [100, 500, 1000, 2000],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.05, 0.1, 0.01],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search_xgb = GridSearchCV(
        estimator=XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            tree_method='hist',   # GPU Parameter
            device='cuda',        # GPU Parameter
            random_state=42
        ),
        param_grid=param_grid_xgb,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,
        cv=5,
        verbose=1
    )

    print("Training XGBoost GridSearch...")
    grid_search_xgb.fit(X_train, y_train)

    print("Best Parameters XGB: ", grid_search_xgb.best_params_)
    XGBmodel_height = grid_search_xgb.best_estimator_

    # cuML Random Forest
    param_grid_rfr = {
        'n_estimators': [100, 500, 1000, 2000],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 7],
        'max_features': ['sqrt', 'log2', 1.0] # 1.0 is safer than None for cuML
    }

    grid_search_rfr = GridSearchCV(
        estimator=cuRFR(),        # cuML Regressor
        param_grid=param_grid_rfr,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,
        verbose=1
    )

    print("Training Random Forest GridSearch...")
    grid_search_rfr.fit(X_train, y_train)

    print("Best Parameters RFC: ", grid_search_rfr.best_params_)
    RFRmodel_height = grid_search_rfr.best_estimator_

else:
    print("\nSimple models training on GPU...")

    # Simple XGBoost
    XGBmodel_height = XGBRegressor(
        n_estimators=200,
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',       # GPU Parameter
        device='cuda',            # GPU Parameter
        early_stopping_rounds=50,
        random_state=42
    )
    XGBmodel_height.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("XGBoost Training finished")

    # Simple cuML Random Forest
    RFRmodel_height = cuRFR(n_estimators=200)
    RFRmodel_height.fit(X_train, y_train)
    print("Random Forest Training finished")


# Evaluation
print("\nEvaluating models on Test Set...")
xgb_pred = XGBmodel_height.predict(X_test)
rfr_pred = RFRmodel_height.predict(X_test)

score_frame = pd.DataFrame({
    "Model": ["XGBRegressor", "RandomForestRegressor"],
    "RMSE": [root_mean_squared_error(y_test, xgb_pred), root_mean_squared_error(y_test, rfr_pred)],
    "MAE": [mean_absolute_error(y_test, xgb_pred), mean_absolute_error(y_test, rfr_pred)]
})

print("\nModel Performance")
print(score_frame.to_string(index=False))

# # Residual Plots
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # XGBoost Residuals
# sns.scatterplot(x=xgb_pred, y=y_test - xgb_pred, ax=axes[0], alpha=0.5, color='blue')
# axes[0].axhline(y=0, color='r', linestyle='--')
# axes[0].set_title("XGBRegressor Residuals")
# axes[0].set_xlabel("Predicted Height")
# axes[0].set_ylabel("Residuals (True - Predicted)")

# # cuML RF Residuals
# sns.scatterplot(x=rfr_pred, y=y_test - rfr_pred, ax=axes[1], alpha=0.5, color='green')
# axes[1].axhline(y=0, color='r', linestyle='--')
# axes[1].set_title("RandomForestRegressor Residuals")
# axes[1].set_xlabel("Predicted Height")
# axes[1].set_ylabel("Residuals (True - Predicted)")

# plt.tight_layout()
# plt.show()

# Saving
xgb_save_path = os.path.join(save_dir, f"{data}XGBmodel_height.joblib")
rfr_save_path = os.path.join(save_dir, f"{data}RFRmodel_height.joblib")

print("\nSaving models...")
joblib.dump(XGBmodel_height, xgb_save_path)
joblib.dump(RFRmodel_height, rfr_save_path)
print(f"Models saved to {save_dir}")