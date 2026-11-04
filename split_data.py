import pandas as pd

from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data/used_cars.csv")

# Calculate car age
reference_year = 2018
df["Car_Age"] = reference_year - df["Year"]

# Select input features
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

# Select the target
y = df["Selling_Price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display the sizes
print("Total records:", len(df))

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining targets:", y_train.shape)
print("Testing targets:", y_test.shape)