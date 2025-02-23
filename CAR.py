import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data
file_path = "stock_data.csv"  # Ensure the correct file path
df = pd.read_csv(file_path, parse_dates=["Date"])

# Convert columns to float
df["Stock_Return"] = df["Stock_Return"].astype(float)
df["Market_Return"] = df["Market_Return"].astype(float)

# Define event window start and end dates
event_start = "2023-01-02"
event_end = "2023-01-10"

# Define estimation window length (e.g., 120 days before event start)
estimation_days = 120
estimation_df = df[df["Date"] < event_start].tail(estimation_days)

# Extract event window data
event_df = df[(df["Date"] >= event_start) & (df["Date"] <= event_end)]

# Regression to estimate alpha and beta
X = sm.add_constant(estimation_df["Market_Return"])
y = estimation_df["Stock_Return"]
model = sm.OLS(y, X).fit()
alpha, beta = model.params

# Compute abnormal returns (AR)
event_df["Expected_Return"] = alpha + beta * event_df["Market_Return"]
event_df["Abnormal_Return"] = event_df["Stock_Return"] - event_df["Expected_Return"]

# Compute cumulative abnormal return (CAR)
event_df["CAR"] = event_df["Abnormal_Return"].cumsum()

# Display results
import ace_tools as tools

tools.display_dataframe_to_user(
    name="Cumulative Abnormal Return (CAR)", dataframe=event_df
)

# Plot CAR
plt.figure(figsize=(10, 5))
plt.plot(
    event_df["Date"],
    event_df["CAR"],
    marker="o",
    linestyle="-",
    label="Cumulative Abnormal Return (CAR)",
)
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("Date")
plt.ylabel("CAR")
plt.title("Cumulative Abnormal Return (CAR) from Jan 2 to Jan 10")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()
