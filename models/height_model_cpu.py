import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import joblib
import os

# Parameter to change
data_folder_path = "/content/drive/MyDrive/Thesis"
save_dir = "/content/drive/MyDrive/Thesis/"
grid_search = True
data = "AEF"  #S2 for sentinal-2 data

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

# plt.figure(figsize=(8, 4))           # Plot about distribution of Target values
# y_train.plot.hist(bins=15)
# plt.title("Distribution of y_train")
# plt.show()

if grid_search:
    print("\nGrid search training...")

    # XGBRegressor
    param_grid_xgb = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.05, 0.1, 0.01],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search_xgb = GridSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror', eval_metric='rmse'),
        param_grid=param_grid_xgb,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        cv=5,
        verbose=1
    )

    print("Training XGBoost GridSearch...")
    grid_search_xgb.fit(X_train, y_train)

    print("Best Parameters XGB: ", grid_search_xgb.best_params_)
    XGBmodel_height = grid_search_xgb.best_estimator_

    # Random Forest Regressor
    param_grid_rfr = {
        'n_estimators': [100, 500, 1000, 2000],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 7],
        'max_features': ['sqrt', 'log2', None]
    }


    grid_search_rfr = GridSearchCV(
        estimator=RandomForestRegressor(n_jobs=-1),
        param_grid=param_grid_rfr,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    print("Training Random Forest GridSearch...")
    grid_search_rfr.fit(X_train, y_train)

    print("Best Parameters RFC: ", grid_search_rfr.best_params_)
    RFRmodel_height = grid_search_rfr.best_estimator_

else:
    # Simple models
    XGBmodel_height = XGBRegressor(n_estimators=200, objective='reg:squarederror', eval_metric='rmse', early_stopping_rounds=50)
    XGBmodel_height.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("XGBoost Training finished")

    RFRmodel_height = RandomForestRegressor(n_estimators=200, n_jobs=-1)
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

print("\nModel Performance:")
print(score_frame.to_string(index=False))

# Save model
xgb_save_path = os.path.join(save_dir, f"{data}XGBmodel_height.joblib")
rfr_save_path = os.path.join(save_dir, f"{data}RFRmodel_height.joblib")

print("\nSaving models...")
joblib.dump(XGBmodel_height, xgb_save_path)
joblib.dump(RFRmodel_height, rfr_save_path)
print(f"Models saved to {save_dir}")