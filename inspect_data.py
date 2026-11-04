import pandas as pd

# Load the dataset
df = pd.read_csv("data/used_cars.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

# 1. Summary of numerical columns
print("\nSummary statistics:")
print(df.describe())

# 2. Unique categories in text columns
print("\nFuel types:")
print(df["Fuel_Type"].value_counts())

print("\nTransmission types:")
print(df["Transmission"].value_counts())

print("\nSeller types:")
print(df["Seller_Type"].value_counts())

# 3. Check the range of manufacturing years
print("\nManufacturing year range:")
print(df["Year"].min(), "to", df["Year"].max())

# 4. Check the selling price range
print("\nSelling price range (in lakhs):")
print(df["Selling_Price"].min(), "to", df["Selling_Price"].max())