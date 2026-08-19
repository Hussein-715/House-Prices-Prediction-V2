# California Housing Price Prediction — Linear Regression (scikit-learn)

An end-to-end supervised regression pipeline that predicts median house prices
across California districts using the classic **California Housing dataset**
(derived from 1990 U.S. Census data). This is the second version of a two-part
project: [version 1](#relation-to-version-1) implemented linear regression
from scratch with NumPy; this version rebuilds the same problem with
production-standard tools — scikit-learn, a proper train/test split, and
richer evaluation metrics.

## Table of contents

- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Results](#results)
- [Project structure](#project-structure)
- [Setup & usage](#setup--usage)
- [Relation to version 1](#relation-to-version-1)
- [Future improvements](#future-improvements)

## Dataset

The [California Housing dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
contains ~20,600 records, one per California census block group, with:

| Feature | Description |
|---|---|
| `longitude`, `latitude` | Geographic location |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms`, `total_bedrooms` | Aggregate room counts |
| `population`, `households` | Block population statistics |
| `median_income` | Median income (in tens of thousands of USD) |
| `ocean_proximity` | Categorical distance-to-ocean label |
| `median_house_value` | **Target** — median house value (USD) |

## Pipeline

1. **Load & explore** — shape, dtypes, summary statistics (`df.info()`, `df.describe()`)
2. **Handle missing values** — numeric columns imputed with the column mean, categorical with the mode; rows missing the target are dropped
3. **Prepare features/target** — `median_house_value` extracted as `y`; non-numeric columns (`ocean_proximity`) excluded from `X` for this version
4. **Train/test split** — 80/20 split via `train_test_split(random_state=42)`
5. **Feature scaling** — `StandardScaler` fit on the training set only, then applied to both sets (prevents test-set leakage into training statistics)
6. **Train** — scikit-learn's `LinearRegression` fit on the scaled training data
7. **Predict** — inference on the held-out test set
8. **Evaluate** — MSE, MAE, and R² computed on test predictions
9. **Visualize** — feature-target relationship, predicted-vs-actual scatter, and target distribution

## Results

Evaluated on the 20% held-out test set:

| Metric | Value |
|---|---|
| MSE | ≈ 5.05 × 10⁹ |
| MAE | ≈ \$51,836 |
| R² | ≈ 0.614 |

An R² of ~0.61 means the model explains about 61% of the variance in house
prices using only numeric features — a reasonable linear baseline, with room
to improve (see below).

## Project structure

```
├── Data/
│   └── California_Housing.csv
├── House_Price_Prediction.ipynb   # full pipeline, steps 1–12
├── requirements.txt
└── .gitignore
```

> The notebook loads the dataset via `Data/California_Housing.csv` — keep the
> capitalized `Data/` folder name so the path resolves without edits.

## Setup & usage

```bash
# clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# launch Jupyter and run the notebook top to bottom
jupyter lab
```

## Relation to version 1

| | v1 — NumPy from scratch | v2 — scikit-learn (this repo) |
|---|---|---|
| Implementation | Manual gradient descent | `LinearRegression` |
| Dataset | Small synthetic housing set | California Housing (~20k rows) |
| Train/test split | None (trained & evaluated on full data) | 80/20 split |
| Metrics | MSE, MAE | MSE, MAE, R² |
| Categorical features | N/A | Dropped (`ocean_proximity`) — planned for encoding |

Building the model manually first, then reproducing it with scikit-learn,
was intentional: it makes clear what the library is actually doing under
the hood rather than treating `.fit()` as a black box.

## Future improvements

- [ ] One-hot encode `ocean_proximity` instead of dropping it
- [ ] Feature engineering (e.g., rooms-per-household, population-per-household)
- [ ] Compare against regularized models (Ridge/Lasso) and non-linear baselines
- [ ] Cross-validation instead of a single train/test split

## License

MIT
