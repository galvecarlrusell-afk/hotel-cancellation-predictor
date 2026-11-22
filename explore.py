import pandas as pd

df = pd.read_csv("data/hotel_reservations.csv")

print("Rows and columns:", df.shape)
print("\nMissing values:", df.isna().sum().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nCancel rate:")
print(df["booking_status"].value_counts(normalize=True))
print("\nSummary of numbers:")
print(df.describe())
print("\nColumn types:")
print(df.dtypes)
print("\nZero adults:", (df["no_of_adults"] == 0).sum())
print("Zero nights:", ((df["no_of_weekend_nights"] + df["no_of_week_nights"]) == 0).sum())
print("Zero price:", (df["avg_price_per_room"] == 0).sum())
