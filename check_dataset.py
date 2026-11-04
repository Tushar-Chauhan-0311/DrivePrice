import pandas as pd

# Load the dataset
df = pd.read_csv("data/used_cars.csv")

# Display basic information
print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nManufacturing Year Range:")
print(df["Year"].min(), "to", df["Year"].max())

print("\nSelling Price Statistics (in lakh):")
print(df["Selling_Price"].describe())

print("\nFirst 5 Records:")
print(df.head())

# Display duplicate records
print("\nDuplicate Records:")

duplicates = df[df.duplicated(keep=False)]

print(duplicates.sort_values(by="Car_Name"))

# Analyze previous owner distribution

print("\nPREVIOUS OWNER DISTRIBUTION")

print(
    df["Owner"]
    .value_counts()
    .sort_index()
)

print("\nAverage selling price by previous owner")

print(
    df.groupby("Owner")["Selling_Price"]
    .agg(["count", "mean", "median"])
    .round(2)
)