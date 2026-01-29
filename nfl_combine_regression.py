"""
Multi-linear regression for NFL Combine 40-yard dash times.

Uses height and weight as input features to predict 40-yard dash time
for wide receivers.
"""

import nfl_data_py as nfl
import pandas as pd
from sklearn.linear_model import LinearRegression


def main():
    # Download all available combine data
    years = list(range(2000, 2026))
    print(f"Downloading combine data for years {years[0]}-{years[-1]}...")
    combine_df = nfl.import_combine_data(years)
    print(f"Total combine records downloaded: {len(combine_df)}")

    # Filter for wide receivers
    wr_df = combine_df[combine_df["pos"] == "WR"].copy()
    print(f"Wide receiver records: {len(wr_df)}")

    # Convert height from "feet-inches" string (e.g. "6-2") to total inches
    def height_to_inches(ht):
        try:
            parts = str(ht).split("-")
            return int(parts[0]) * 12 + int(parts[1])
        except (ValueError, IndexError):
            return None

    wr_df["ht"] = wr_df["ht"].apply(height_to_inches)

    # Keep only rows where height, weight, and 40-yard dash are all present
    required_cols = ["ht", "wt", "forty"]
    wr_complete = wr_df.dropna(subset=required_cols).copy()
    print(f"WR records with height, weight, and 40yd dash: {len(wr_complete)}")

    # Prepare features and target
    X = wr_complete[["ht", "wt"]]
    y = wr_complete["forty"]

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Display results
    print("\n=== Multi-Linear Regression Results ===")
    print(f"Target: 40-yard dash time (seconds)")
    print(f"Features: height (inches), weight (lbs)")
    print(f"Samples used: {len(wr_complete)}")
    print()
    print(f"Intercept:            {model.intercept_:.6f}")
    print(f"Coefficient (height): {model.coef_[0]:.6f}")
    print(f"Coefficient (weight): {model.coef_[1]:.6f}")
    print(f"R² score:             {model.score(X, y):.6f}")
    print()
    print("Equation:")
    print(
        f"  40yd = {model.intercept_:.4f} "
        f"+ ({model.coef_[0]:.4f} × height) "
        f"+ ({model.coef_[1]:.4f} × weight)"
    )


if __name__ == "__main__":
    main()
