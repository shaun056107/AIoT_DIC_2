import os
import json
import matplotlib.pyplot as plt

def generate_static_charts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "data", "precomputed.json")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for ds_key, title_name in [("startups", "50 Startups"), ("boston", "Boston Housing")]:
        ds = data[ds_key]
        methods = ds["methods"]
        k_vals = [pk["k"] for pk in methods[0]["per_k"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left: RMSE
        for m in methods:
            rmse_vals = [pk["rmse"] for pk in m["per_k"]]
            lw = 3.0 if m["id"] == "exh" else 1.8
            marker = "*" if m["id"] == "exh" else "o"
            ms = 10 if m["id"] == "exh" else 6
            ax1.plot(k_vals, rmse_vals, label=m["name"], color=m["color"], 
                     linewidth=lw, marker=marker, markersize=ms)

        ax1.set_title(f"RMSE Comparison across 5 Feature Selection Methods\n({title_name})", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of Features", fontsize=11)
        ax1.set_ylabel("RMSE (Lower is Better)", fontsize=11)
        ax1.set_xticks(k_vals)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="best", frameon=True)

        # Right: R-squared
        for m in methods:
            r2_vals = [pk["r2"] for pk in m["per_k"]]
            lw = 3.0 if m["id"] == "exh" else 1.8
            marker = "*" if m["id"] == "exh" else "o"
            ms = 10 if m["id"] == "exh" else 6
            ax2.plot(k_vals, r2_vals, label=m["name"], color=m["color"], 
                     linewidth=lw, marker=marker, markersize=ms)

        ax2.set_title(f"R-squared Comparison across 5 Feature Selection Methods\n({title_name})", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Features", fontsize=11)
        ax2.set_ylabel("R-squared (Higher is Better)", fontsize=11)
        ax2.set_xticks(k_vals)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="best", frameon=True)

        plt.tight_layout()
        out_png = os.path.join(plots_dir, f"{ds_key}_comparison.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Generated chart: {out_png}")

if __name__ == "__main__":
    generate_static_charts()
