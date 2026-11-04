import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load our dataset
df = pd.read_csv("data/used_cars.csv")

sns.set_theme(style="whitegrid")

# Chart 1: Distribution of selling prices
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Selling_Price", bins=20, kde=True)
plt.title("Distribution of Used Car Selling Prices")
plt.xlabel("Selling Price (in lakhs)")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()

# Chart 2: Present price vs selling price
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Present_Price",
    y="Selling_Price"
)
plt.title("Present Price vs Selling Price")
plt.xlabel("Present Price (in lakhs)")
plt.ylabel("Selling Price (in lakhs)")
plt.tight_layout()
plt.show()

# Chart 3: Check unusually high mileage
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="Kms_Driven")
plt.title("Distribution of Kilometers Driven")
plt.xlabel("Kilometers Driven")
plt.tight_layout()
plt.show()

