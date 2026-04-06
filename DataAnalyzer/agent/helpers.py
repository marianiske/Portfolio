import os
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import difflib
DATASETS = {
            'Fraud detection': 
                {
                    'slug': 'dhruvb2028/credit-card-fraud-dataset',
                     'csv_name': 'credit_card_frauds.csv'
                 },
            'House prices':
                {
                    'slug':'pcbreviglieri/house-prices',
                    'csv_name':'train.csv'
                },
            'Productivity':
                {
                    'slug': 'asifxzaman/social-media-addiction-vs-productivity-dataset',
                    'csv_name': 'social_media_productivity_6000.csv'
                }
            }

def find_dataset(query: str) -> dict:
    best_ds = find_best_dataset_match(query)
    if best_ds is None:
        return {"match_found": False, "dataset_name": None, 'data': {}}
    return {"match_found": True, "dataset_name": best_ds}
    

def find_best_dataset_match(query: str) -> str | None:
    names = list(DATASETS.keys())
    lower_map = {name.lower(): name for name in names}

    query_lower = query.lower()

    for name in names:
        if query_lower in name.lower() or any(word in name.lower() for word in query_lower.split()):
            return name

    matches = difflib.get_close_matches(query_lower, lower_map.keys(), n=1, cutoff=0.4)
    if matches:
        return lower_map[matches[0]]

    return None
    
def list_datasets() -> dict:
    return {
        "available_datasets": list(DATASETS.keys())
    }

def load_dataset_by_name(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    dataset_slug = DATASETS[dataset_name]["slug"]
    csv_filename = DATASETS[dataset_name]["csv_name"]
    return load_dataset(dataset_slug, csv_filename)

def load_dataset(dataset_slug: str, csv_filename: str) -> pd.DataFrame:
    dataset_path = kagglehub.dataset_download(dataset_slug)
    csv_path = os.path.join(dataset_path, csv_filename)
    return pd.read_csv(csv_path)
    
def calc_mean_column(df: pd.DataFrame, col: str) -> dict:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    return {"column": col, "mean": float(df[col].mean())}


def get_info(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    return {
        "rows": int(rows),
        "columns": int(cols),
        "column_names": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": {col: int(v) for col, v in df.isna().sum().items()},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def describe_dataset(df: pd.DataFrame) -> dict:
    desc = df.describe(include="all").fillna("")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "summary_preview": desc.to_dict()
    }

def scatter_plot(df: pd.DataFrame, col1: str, col2: str,
                 plot_description: str = "", linewidths: float = 0.7, output_path='/tmp') -> dict:
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns '{col1}' and/or '{col2}' not found.")

    plt.figure(figsize=(6, 4))
    plt.scatter(df[col1], df[col2], linewidths=linewidths)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(plot_description or f"{col1} vs {col2}")

    output_path = output_path + f"/scatter_{col1}_{col2}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return {
        "plot_created": True,
        "plot_path": output_path,
        "x": col1,
        "y": col2,
        "title": plot_description or f"{col1} vs {col2}",
    }


def pca_feature_selection(df: pd.DataFrame, n_features: int = 2) -> dict:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise ValueError("Dataset has no numeric columns for PCA.")

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(numeric_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=n_features)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(n_features)],
        index=df.index,
    )

    return {
        "n_features": int(n_features),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "components_preview": pca_df.head(10).to_dict(orient="records"),
    }


def filter_outliers(df: pd.DataFrame, threshold: float = 3.5) -> dict:
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return {
            "rows_before": int(len(df)),
            "rows_after": int(len(df)),
            "rows_removed": 0,
            "note": "No numeric columns found."
        }

    keep_mask = pd.Series(True, index=result.index)

    for col in numeric_cols:
        x = result[col]
        non_na = x.dropna()

        if non_na.empty:
            continue

        median = non_na.median()
        mad = np.median(np.abs(non_na - median))

        if mad == 0:
            continue

        robust_z = 0.6745 * (x - median) / mad
        outliers = robust_z.abs() > threshold
        outliers = outliers.fillna(False)

        keep_mask &= ~outliers

    filtered = result.loc[keep_mask].copy()

    return {
        "rows_before": int(len(df)),
        "rows_after": int(len(filtered)),
        "rows_removed": int(len(df) - len(filtered)),
        "threshold": float(threshold),
        "preview": filtered.head(10).fillna("").to_dict(orient="records"),
    }


def normalize(df: pd.DataFrame) -> dict:
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns

    result[numeric_cols] = (
        result[numeric_cols] - result[numeric_cols].mean()
    ) / result[numeric_cols].std().replace(0, 1)

    return {
        "normalized_columns": list(numeric_cols),
        "preview": result.head(10).fillna("").to_dict(orient="records"),
    }


def missing_values_report(df: pd.DataFrame) -> dict:
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().sum() / len(df) * 100).round(2),
        "dtype": df.dtypes.astype(str)
    }).sort_values("missing_count", ascending=False)

    return {"report": report.reset_index(names="column").to_dict(orient="records")}


def column_type_report(df: pd.DataFrame) -> dict:
    report = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_unique": [int(df[col].nunique(dropna=True)) for col in df.columns],
        "example_value": [
            None if df[col].dropna().empty else str(df[col].dropna().iloc[0])
            for col in df.columns
        ]
    })
    return {"report": report.to_dict(orient="records")}


def correlation_report(df: pd.DataFrame, method: str = "pearson") -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("Dataset has no numeric columns for correlation analysis.")
    corr = numeric_df.corr(method=method)
    return {
        "method": method,
        "columns": list(corr.columns),
        "correlation_matrix": corr.round(4).to_dict(orient="index"),
    }


def categorical_summary(df: pd.DataFrame, top_n: int = 10) -> dict:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    summary = {}

    for col in cat_cols:
        summary[col] = df[col].value_counts(dropna=False).head(top_n).to_dict()

    return {"top_n": int(top_n), "summary": summary}


def duplicate_report(df: pd.DataFrame) -> dict:
    dup_count = int(df.duplicated().sum())
    return {
        "duplicate_rows": dup_count,
        "duplicate_percent": round(dup_count / len(df) * 100, 2) if len(df) > 0 else 0.0
    }