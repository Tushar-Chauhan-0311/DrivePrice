import pandas as pd

# Load the dataset
df = pd.read_csv("data/used_cars.csv")

# Create a copy so the original data stays unchanged
df = df.copy()

# Calculate the age of each car
reference_year = 2018
df["Car_Age"] = reference_year - df["Year"]

# Select the features our model will use
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

# Select the target we want to predict
y = df["Selling_Price"]

# Display the results
print("Features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)