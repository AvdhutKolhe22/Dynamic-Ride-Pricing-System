# =========================================
# DYNAMIC RIDE PRICING SYSTEM
# =========================================

# -------------------------------
# IMPORT LIBRARIES
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("../dataset/dynamic_pricing.csv")

# =========================================
# BASIC DATA UNDERSTANDING
# =========================================

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== COLUMN NAMES ==========\n")
print(df.columns)

print("\n========== DATA TYPES ==========\n")
print(df.dtypes)

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

print("\n========== NUMERICAL SUMMARY ==========\n")
print(df.describe())

# =========================================
# TARGET VARIABLE SUMMARY
# =========================================

print("\n========== TARGET VARIABLE SUMMARY ==========\n")
print(df["Historical_Cost_of_Ride"].describe())

# =========================================
# CREATE VISUALS
# =========================================

# HISTOGRAM

plt.figure(figsize=(8,5))

sns.histplot(
    df["Historical_Cost_of_Ride"],
    kde=True
)

plt.title("Distribution of Ride Cost")
plt.xlabel("Ride Cost")
plt.ylabel("Frequency")

plt.savefig("../visuals/ride_cost_distribution.png")

plt.close()

# =========================================
# RIDERS VS COST
# =========================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x=df["Number_of_Riders"],
    y=df["Historical_Cost_of_Ride"]
)

plt.title("Number of Riders vs Ride Cost")
plt.xlabel("Number of Riders")
plt.ylabel("Historical Cost of Ride")

plt.savefig("../visuals/riders_vs_cost.png")

plt.close()

# =========================================
# DRIVERS VS COST
# =========================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x=df["Number_of_Drivers"],
    y=df["Historical_Cost_of_Ride"]
)

plt.title("Number of Drivers vs Ride Cost")
plt.xlabel("Number of Drivers")
plt.ylabel("Historical Cost of Ride")

plt.savefig("../visuals/drivers_vs_cost.png")

plt.close()

# =========================================
# VEHICLE TYPE VS COST
# =========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["Vehicle_Type"],
    y=df["Historical_Cost_of_Ride"]
)

plt.title("Vehicle Type vs Ride Cost")

plt.savefig("../visuals/vehicle_type_vs_cost.png")

plt.close()

# =========================================
# TIME OF BOOKING VS COST
# =========================================

plt.figure(figsize=(10,5))

sns.boxplot(
    x=df["Time_of_Booking"],
    y=df["Historical_Cost_of_Ride"]
)

plt.title("Time of Booking vs Ride Cost")

plt.xticks(rotation=15)

plt.savefig("../visuals/time_vs_cost.png")

plt.close()

# =========================================
# CORRELATION HEATMAP
# =========================================

plt.figure(figsize=(10,6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("../visuals/heatmap.png")

plt.close()

print("\n========== EDA COMPLETED ==========\n")

# =========================================
# ENCODE CATEGORICAL COLUMNS
# =========================================

categorical_columns = [
    "Location_Category",
    "Customer_Loyalty_Status",
    "Time_of_Booking",
    "Vehicle_Type"
]

encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# =========================================
# FEATURES & TARGET
# =========================================

X = df.drop("Historical_Cost_of_Ride", axis=1)

y = df["Historical_Cost_of_Ride"]

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# MACHINE LEARNING MODELS
# =========================================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

best_model = None
best_r2 = 0
best_model_name = ""

# =========================================
# TRAINING & EVALUATION
# =========================================

for name, model in models.items():

    print(f"\n========== {name} ==========\n")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    print("Mean Absolute Error:", mae)

    print("R2 Score:", r2)

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# =========================================
# BEST MODEL
# =========================================

print("\n========== BEST MODEL ==========\n")

print("Best Model:", best_model_name)

print("Best R2 Score:", best_r2)

# =========================================
# DYNAMIC PRICE PREDICTION
# =========================================

print("\n========== DYNAMIC PRICE PREDICTION ==========\n")

new_ride = pd.DataFrame({
    "Number_of_Riders": [80],
    "Number_of_Drivers": [20],
    "Location_Category": [2],
    "Customer_Loyalty_Status": [1],
    "Number_of_Past_Rides": [50],
    "Average_Ratings": [4.5],
    "Time_of_Booking": [2],
    "Vehicle_Type": [1],
    "Expected_Ride_Duration": [120]
})

predicted_price = best_model.predict(new_ride)

print("Recommended Dynamic Ride Price:")

print(predicted_price[0])

print("\n========== PROJECT COMPLETED ==========\n")