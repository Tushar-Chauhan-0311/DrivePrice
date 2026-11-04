import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_csv("data/used_cars.csv")

# Remove exact duplicate records
df = df.drop_duplicates().reset_index(drop=True)

print("Dataset shape after removing duplicates:", df.shape)

print("=" * 55)
print("          DRIVEPRICE MODEL TRAINING")
print("=" * 55)

print(f"\nDataset loaded successfully: {df.shape[0]} records")


# ==========================================
# STEP 2: FEATURE ENGINEERING
# ==========================================

# The dataset contains historical car listings.
# We retain 2018 as the reference year to match
# the existing training setup.

reference_year = 2018

df["Car_Age"] = reference_year - df["Year"]


# ==========================================
# STEP 3: SELECT FEATURES AND TARGET
# ==========================================

X = df[
    [
        "Car_Age",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner",
    ]
]

y = df["Selling_Price"]


# ==========================================
# STEP 4: SPLIT THE DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")


# ==========================================
# STEP 5: PREPROCESSING
# ==========================================

categorical_columns = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
]

numerical_columns = [
    "Car_Age",
    "Present_Price",
    "Kms_Driven",
    "Owner",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns,
        ),
    ]
)


# ==========================================
# STEP 6: DEFINE MODELS
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=3,
        random_state=42,
    ),

    "Random Forest": RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
),
}

# ==========================================
# ==========================================
# STEP 6.1: CROSS-VALIDATION
# ==========================================

print("\n" + "=" * 65)
print("             CROSS-VALIDATION RESULTS")
print("=" * 65)

cv_results = []

for model_name, regressor in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    cv_strategy = RepeatedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

    scores = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv_strategy,
    scoring={
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    },
    n_jobs=1,
)

    mean_mae = -scores["test_mae"].mean()
    mean_r2 = scores["test_r2"].mean()

    cv_results.append(
        {
            "Model": model_name,
            "CV_MAE": mean_mae,
            "CV_R2": mean_r2,
        }
    )

    print(f"\n{model_name}")
    print(f"Average CV MAE: {mean_mae:.3f} lakh")
    print(f"Average CV R²:  {mean_r2:.3f}")

# Store cross-validation results
cv_results_df = pd.DataFrame(cv_results)

# ==========================================
# STEP 7: TRAIN AND EVALUATE EACH MODEL
# ==========================================

results = []
trained_models = {}

for model_name, regressor in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on unseen test data
    predictions = pipeline.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append(
        {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }
    )

    trained_models[model_name] = pipeline

    print(f"MAE:  {mae:.3f} lakh")
    print(f"RMSE: {rmse:.3f} lakh")
    print(f"R²:   {r2:.3f}")


# ==========================================
# STEP 8: COMPARE MODELS
# ==========================================

# Combine cross-validation and test results
results_df = pd.DataFrame(results)

comparison_df = cv_results_df.merge(
    results_df,
    on="Model",
)

# Select the model using cross-validation MAE
comparison_df = comparison_df.sort_values(
    by="CV_MAE",
    ascending=True,
).reset_index(drop=True)

print("\n" + "=" * 85)
print("                         MODEL COMPARISON")
print("=" * 85)

print(
    comparison_df.to_string(
        index=False,
        formatters={
            "CV_MAE": "{:.3f}".format,
            "CV_R2": "{:.3f}".format,
            "MAE": "{:.3f}".format,
            "RMSE": "{:.3f}".format,
            "R2": "{:.3f}".format,
        },
    )
)

# ==========================================
# STEP 9: SELECT THE BEST MODEL
# ==========================================

# Select the best model using cross-validation MAE
best_model_name = comparison_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

# Check feature importance
print("\nFEATURE IMPORTANCE")

feature_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = best_model.named_steps[
    "regressor"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.to_string(index=False))

best_result = comparison_df.iloc[0]

print("\n" + "=" * 55)
print(f"Best model by cross-validation MAE: {best_model_name}")
print(f"Average CV MAE: {best_result['CV_MAE']:.3f} lakh")
print("\nFinal evaluation on the test set:")
print(f"Test MAE: {best_result['MAE']:.3f} lakh")
print(f"Test RMSE: {best_result['RMSE']:.3f} lakh")
print(f"Test R²: {best_result['R2']:.3f}")


# Permutation Feature Importance

print("\nPERMUTATION FEATURE IMPORTANCE")

permutation = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error"
)

permutation_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": permutation.importances_mean
})

permutation_df = permutation_df.sort_values(
    by="Importance",
    ascending=False
)

print(permutation_df.to_string(index=False))

# ==========================================
# STEP 11: PREDICTION ERROR ANALYSIS
# ==========================================

print("\n" + "=" * 65)
print("             PREDICTION ERROR ANALYSIS")
print("=" * 65)

# Generate predictions using the selected model
best_predictions = best_model.predict(X_test)

# Create a comparison table with original car features
error_analysis = X_test.copy()

# Add car name and manufacturing year for better analysis
error_analysis.insert(
    0, "Car_Name", df.loc[X_test.index, "Car_Name"]
)

error_analysis.insert(
    1, "Year", df.loc[X_test.index, "Year"]
)

error_analysis["Actual Price (lakh)"] = y_test
error_analysis["Predicted Price (lakh)"] = best_predictions

# Calculate absolute prediction error
error_analysis["Absolute Error (lakh)"] = abs(
    error_analysis["Actual Price (lakh)"]
    - error_analysis["Predicted Price (lakh)"]
)

# Sort predictions by absolute error
error_analysis = error_analysis.sort_values(
    by="Absolute Error (lakh)",
    ascending=False,
)

# Display the 10 predictions with the largest errors
print("\n10 predictions with the largest errors:")
print(error_analysis.head(10).round(3).to_string())

# Calculate additional error statistics
print("\nError summary:")
print(
    f"Average absolute error: "
    f"{error_analysis['Absolute Error (lakh)'].mean():.3f} lakh"
)

print(
    f"Median absolute error: "
    f"{error_analysis['Absolute Error (lakh)'].median():.3f} lakh"
)

print(
    f"Largest absolute error: "
    f"{error_analysis['Absolute Error (lakh)'].max():.3f} lakh"
)

# Check the distribution of actual prices in the test set
print("\nTEST SET PRICE ANALYSIS")
print("Minimum actual price:", y_test.min(), "lakh")
print("Maximum actual price:", y_test.max(), "lakh")
print("Average actual price:", y_test.mean(), "lakh")
print("Median actual price:", y_test.median(), "lakh")

print("\nNumber of test cars priced above 15 lakh:")
print((y_test > 15).sum())


# Check the effect of previous ownership on prediction

print("\nOWNER IMPACT TEST")

sample_car = X_test.iloc[[0]].copy()

for owner_value in [0, 1, 2, 3]:
    sample_car["Owner"] = owner_value

    prediction = best_model.predict(sample_car)[0]

    print(
        f"Previous owners: {owner_value} "
        f"-> Predicted price: ₹{prediction:.4f} lakh"
    )

# ==========================================
# STEP 10: SAVE BEST MODEL
# ==========================================

joblib.dump(best_model, "car_price_model.pkl")

print("\nBest model saved as car_price_model.pkl")
print("Training and evaluation completed!")
print("=" * 55)