"""
Composite athleticism score for NFL Combine wide receivers.

Combines weight, 3-cone drill time, and 40-yard dash time into a single
normalized score. Each metric is z-score normalized and then averaged,
with sign flips so that higher composite = better athlete:
  - Weight: higher is better (more size/physicality)
  - 3-cone drill: lower is better (more agile), so z-score is negated
  - 40-yard dash: lower is better (faster), so z-score is negated
"""

import nfl_data_py as nfl
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    # Download all available combine data
    years = list(range(2000, 2026))
    print(f"Downloading combine data for years {years[0]}-{years[-1]}...")
    combine_df = nfl.import_combine_data(years)
    print(f"Total combine records downloaded: {len(combine_df)}")

    # Filter for wide receivers
    wr_df = combine_df[combine_df["pos"] == "WR"].copy()
    print(f"Wide receiver records: {len(wr_df)}")

    # Keep only rows where weight, 3-cone drill, and 40-yard dash are all present
    required_cols = ["wt", "cone", "forty"]
    wr_complete = wr_df.dropna(subset=required_cols).copy()
    print(f"WR records with weight, 3-cone, and 40yd: {len(wr_complete)}")

    # Normalize the three metrics
    scaler = StandardScaler()
    scaled = scaler.fit_transform(wr_complete[required_cols])

    # Build composite: +weight, -three_cone, -forty
    # Higher composite = heavier + more agile + faster
    z_wt = scaled[:, 0]
    z_three_cone = scaled[:, 1]
    z_forty = scaled[:, 2]
    wr_complete["composite"] = (z_wt - z_three_cone - z_forty) / 3

    # Display normalization parameters
    print(f"\nFeature means:   weight={scaler.mean_[0]:.2f} lbs, "
          f"3-cone={scaler.mean_[1]:.4f} s, 40yd={scaler.mean_[2]:.4f} s")
    print(f"Feature stddevs: weight={scaler.scale_[0]:.2f} lbs, "
          f"3-cone={scaler.scale_[1]:.4f} s, 40yd={scaler.scale_[2]:.4f} s")
    print(f"\nFormula: composite = (z_weight - z_three_cone - z_forty) / 3")
    print("Higher composite = heavier + more agile + faster")

    # Sort by composite score descending
    wr_ranked = wr_complete.sort_values("composite", ascending=False).reset_index(drop=True)

    # Display top 20
    print("\n=== Top 20 WRs by Composite Score ===")
    print(f"{'Rank':<6}{'Player':<28}{'Year':<6}{'Weight':<10}{'3-Cone':<10}{'40yd':<10}{'Composite'}")
    print("-" * 86)
    for i, row in wr_ranked.head(20).iterrows():
        print(f"{i+1:<6}{row['player_name']:<28}{int(row['season']):<6}"
              f"{row['wt']:<10.0f}{row['cone']:<10.2f}{row['forty']:<10.2f}"
              f"{row['composite']:.4f}")

    # Display bottom 10
    print(f"\n=== Bottom 10 WRs by Composite Score ===")
    print(f"{'Rank':<6}{'Player':<28}{'Year':<6}{'Weight':<10}{'3-Cone':<10}{'40yd':<10}{'Composite'}")
    print("-" * 86)
    total = len(wr_ranked)
    for i, row in wr_ranked.tail(10).iterrows():
        rank = total - (len(wr_ranked) - 1 - i)
        print(f"{rank:<6}{row['player_name']:<28}{int(row['season']):<6}"
              f"{row['wt']:<10.0f}{row['cone']:<10.2f}{row['forty']:<10.2f}"
              f"{row['composite']:.4f}")

    # Summary stats
    print(f"\n=== Composite Score Distribution ===")
    print(f"Mean:   {wr_ranked['composite'].mean():.4f}")
    print(f"Std:    {wr_ranked['composite'].std():.4f}")
    print(f"Min:    {wr_ranked['composite'].min():.4f}")
    print(f"Max:    {wr_ranked['composite'].max():.4f}")
    print(f"Median: {wr_ranked['composite'].median():.4f}")


if __name__ == "__main__":
    main()
