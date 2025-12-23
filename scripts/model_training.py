import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_PATH = "C:\movie_project"

DATA_PATH = os.path.join(BASE_PATH, "data", "movies_cleaned.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")

os.makedirs(OUTPUT_PATH, exist_ok=True)


df = pd.read_csv(DATA_PATH)


features = [
    "Audience score %",
    "Rotten Tomatoes %",
    "Worldwide Gross"
]

target = "Profitability"

df_model = df[features + [target]].dropna()

X = df_model[features]
y = df_model[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


importance = model.feature_importances_


result_file = os.path.join(OUTPUT_PATH, "model_results.txt")

with open(result_file, "w") as f:
    f.write("Movie Success Prediction - Model Results\n")
    f.write("----------------------------------------\n\n")
    f.write("Model Used: Random Forest Regressor\n\n")
    f.write("Features Used:\n")
    for feat in features:
        f.write(f"- {feat}\n")

    f.write("\nEvaluation Metrics:\n")
    f.write(f"MAE  : {mae:.4f}\n")
    f.write(f"RMSE : {rmse:.4f}\n")
    f.write(f"R2   : {r2:.4f}\n\n")

    f.write("Feature Importance:\n")
    for feat, imp in zip(features, importance):
        f.write(f"{feat}: {imp:.4f}\n")

print("✅ Model training completed successfully!")
print("📁 Results saved to:", result_file)
