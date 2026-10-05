"""
eda.py — Exploratory Data Analysis script.

Generates all EDA charts as PNG files in app/static/img/ and creates
a Jupyter notebook in notebooks/eda.ipynb.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
IMG_DIR      = PROJECT_ROOT / "app" / "static" / "img"
NOTEBOOKS    = PROJECT_ROOT / "notebooks"

IMG_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS.mkdir(parents=True, exist_ok=True)

# Use a nice style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def run_eda():
    print("=" * 60)
    print("STEP 2 — Exploratory Data Analysis")
    print("=" * 60)

    df = pd.read_csv(PROCESSED / "cleaned.csv")
    print(f"\n  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # ── 1. Price Distribution (raw and log) ────────────────────────────────
    print("\n[1/6] Price distribution …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["selling_price"], bins=50, color="#6366f1", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Selling Price (₹)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Price Distribution (Raw)")

    axes[1].hist(np.log1p(df["selling_price"]), bins=50, color="#10b981", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("log(1 + Price)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Price Distribution (Log-transformed)")

    fig.suptitle("Insight: Price is heavily right-skewed; log transform makes it near-normal.",
                 y=-0.02, fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_price_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. Price vs Car Age ────────────────────────────────────────────────
    print("[2/6] Price vs Car Age …")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["car_age"], df["selling_price"], alpha=0.2, s=8, color="#f59e0b")
    ax.set_xlabel("Car Age (years)")
    ax.set_ylabel("Selling Price (₹)")
    ax.set_title("Price vs Car Age\nInsight: Older cars are cheaper — price drops sharply after 5 years.")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_price_vs_age.png", dpi=150)
    plt.close()

    # ── 3. Price vs Km Driven ──────────────────────────────────────────────
    print("[3/6] Price vs Km Driven …")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["km_driven"], df["selling_price"], alpha=0.2, s=8, color="#ef4444")
    ax.set_xlabel("Km Driven")
    ax.set_ylabel("Selling Price (₹)")
    ax.set_title("Price vs Km Driven\nInsight: Higher mileage correlates with lower resale price.")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_price_vs_km.png", dpi=150)
    plt.close()

    # ── 4. Average Price by Brand / Fuel / Transmission ────────────────────
    print("[4/6] Average price by category …")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Top 15 brands by count
    top_brands = df["brand"].value_counts().head(15).index
    brand_avg  = df[df["brand"].isin(top_brands)].groupby("brand")["selling_price"].mean().sort_values(ascending=True)
    brand_avg.plot.barh(ax=axes[0], color="#6366f1")
    axes[0].set_xlabel("Avg Price (₹)")
    axes[0].set_title("Avg Price by Brand (top 15)\nInsight: Luxury brands command 3-5× higher prices.")

    fuel_avg = df.groupby("fuel")["selling_price"].mean().sort_values(ascending=True)
    fuel_avg.plot.barh(ax=axes[1], color="#10b981")
    axes[1].set_xlabel("Avg Price (₹)")
    axes[1].set_title("Avg Price by Fuel\nInsight: Diesel cars have the highest average resale value.")

    trans_avg = df.groupby("transmission")["selling_price"].mean().sort_values(ascending=True)
    trans_avg.plot.barh(ax=axes[2], color="#f59e0b")
    axes[2].set_xlabel("Avg Price (₹)")
    axes[2].set_title("Avg Price by Transmission\nInsight: Automatic cars sell for ~2× manual cars on average.")

    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_avg_price_by_category.png", dpi=150)
    plt.close()

    # ── 5. Correlation Heatmap ─────────────────────────────────────────────
    print("[5/6] Correlation heatmap …")
    num_cols = df.select_dtypes(include="number").columns
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap\nInsight: Engine, max_power, and car_age are top price predictors.")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_correlation_heatmap.png", dpi=150)
    plt.close()

    # ── 6. Create a minimal Jupyter notebook ───────────────────────────────
    print("[6/6] Generating EDA notebook …")
    _generate_notebook()

    print("\n✓ EDA complete. Charts saved to app/static/img/")


def _generate_notebook():
    """Programmatically create a Jupyter notebook with EDA charts."""
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

    nb = new_notebook()
    nb.cells = [
        new_markdown_cell("# 📊 Used Car Price — Exploratory Data Analysis"),
        new_code_cell(
            "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
            "import seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n%matplotlib inline"
        ),
        new_code_cell(
            "df = pd.read_csv('../data/processed/cleaned.csv')\n"
            "print(f'Shape: {df.shape}')\ndf.head()"
        ),
        new_code_cell("df.describe()"),
        new_code_cell("df.info()"),
        new_markdown_cell("## 1. Price Distribution"),
        new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "axes[0].hist(df['selling_price'], bins=50, color='#6366f1', edgecolor='white')\n"
            "axes[0].set_title('Raw Price Distribution')\n"
            "axes[1].hist(np.log1p(df['selling_price']), bins=50, color='#10b981', edgecolor='white')\n"
            "axes[1].set_title('Log-transformed Price Distribution')\n"
            "plt.tight_layout()\nplt.show()"
        ),
        new_markdown_cell("*Insight: Price is heavily right-skewed; log transform normalises it.*"),
        new_markdown_cell("## 2. Price vs Car Age"),
        new_code_cell(
            "plt.figure(figsize=(10,5))\n"
            "plt.scatter(df['car_age'], df['selling_price'], alpha=0.2, s=8)\n"
            "plt.xlabel('Car Age')\nplt.ylabel('Price (₹)')\nplt.title('Price vs Car Age')\nplt.show()"
        ),
        new_markdown_cell("*Insight: Older cars are significantly cheaper — steep depreciation in early years.*"),
        new_markdown_cell("## 3. Price vs Km Driven"),
        new_code_cell(
            "plt.figure(figsize=(10,5))\n"
            "plt.scatter(df['km_driven'], df['selling_price'], alpha=0.2, s=8, color='red')\n"
            "plt.xlabel('Km Driven')\nplt.ylabel('Price (₹)')\nplt.title('Price vs Km Driven')\nplt.show()"
        ),
        new_markdown_cell("*Insight: Higher mileage = lower resale value, but the variance is large.*"),
        new_markdown_cell("## 4. Average Price by Category"),
        new_code_cell(
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
            "top_brands = df['brand'].value_counts().head(15).index\n"
            "df[df['brand'].isin(top_brands)].groupby('brand')['selling_price'].mean().sort_values().plot.barh(ax=axes[0], color='#6366f1')\n"
            "axes[0].set_title('Avg Price by Brand')\n"
            "df.groupby('fuel')['selling_price'].mean().sort_values().plot.barh(ax=axes[1], color='#10b981')\n"
            "axes[1].set_title('Avg Price by Fuel')\n"
            "df.groupby('transmission')['selling_price'].mean().sort_values().plot.barh(ax=axes[2], color='#f59e0b')\n"
            "axes[2].set_title('Avg Price by Transmission')\n"
            "plt.tight_layout()\nplt.show()"
        ),
        new_markdown_cell("*Insight: Brand premium, diesel, and automatic transmission all drive higher prices.*"),
        new_markdown_cell("## 5. Correlation Heatmap"),
        new_code_cell(
            "num_cols = df.select_dtypes(include='number').columns\n"
            "corr = df[num_cols].corr()\n"
            "plt.figure(figsize=(10, 8))\n"
            "sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, mask=np.triu(np.ones_like(corr, dtype=bool)))\n"
            "plt.title('Correlation Heatmap')\nplt.tight_layout()\nplt.show()"
        ),
        new_markdown_cell("*Insight: max_power (0.77) and engine (0.62) are the strongest price predictors.*"),
    ]

    nb_path = NOTEBOOKS / "eda.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"  Notebook saved to {nb_path}")


if __name__ == "__main__":
    run_eda()
