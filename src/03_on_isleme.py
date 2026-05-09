from pathlib import Path
import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "outputs" / "results"


def fill_missing_values(all_data: pd.DataFrame) -> pd.DataFrame:
    none_columns = (
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
    )
    for column in none_columns:
        if column in all_data.columns:
            all_data[column] = all_data[column].fillna("None")

    zero_columns = (
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "MasVnrArea",
    )
    for column in zero_columns:
        if column in all_data.columns:
            all_data[column] = all_data[column].fillna(0)

    if "LotFrontage" in all_data.columns:
        all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
            lambda values: values.fillna(values.median())
        )

    mode_columns = (
        "MSZoning",
        "Electrical",
        "KitchenQual",
        "Exterior1st",
        "Exterior2nd",
        "SaleType",
        "Utilities",
        "Functional",
    )
    for column in mode_columns:
        if column in all_data.columns:
            all_data[column] = all_data[column].fillna(all_data[column].mode()[0])

    return all_data


def encode_ordinal_features(all_data: pd.DataFrame) -> pd.DataFrame:
    quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    quality_columns = (
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
    )
    for column in quality_columns:
        if column in all_data.columns:
            all_data[column] = all_data[column].map(quality_map).fillna(0).astype(int)

    ordinal_maps = {
        "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
        "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
        "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
        "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
        "PavedDrive": {"N": 0, "P": 1, "Y": 2},
        "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
        "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    }

    for column, mapping in ordinal_maps.items():
        if column in all_data.columns:
            all_data[column] = all_data[column].map(mapping).fillna(0).astype(int)

    return all_data


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")

    original_train_rows = train.shape[0]
    outlier_mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
    train = train.drop(train[outlier_mask].index)
    removed_rows = original_train_rows - train.shape[0]

    y_train = np.log1p(train["SalePrice"])
    y_train.name = "SalePrice_log"

    train = train.drop(columns=["SalePrice", "Id"])
    test = test.drop(columns=["Id"])
    n_train = train.shape[0]

    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

    all_data = fill_missing_values(all_data)

    if "MSSubClass" in all_data.columns:
        all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)

    all_data = encode_ordinal_features(all_data)
    all_data = pd.get_dummies(all_data)

    numeric_features = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_features = all_data[numeric_features].apply(lambda values: values.skew()).sort_values(ascending=False)
    high_skew = skewed_features[abs(skewed_features) > 0.75]

    for feature in high_skew.index:
        all_data[feature] = np.log1p(all_data[feature])

    x_train = all_data.iloc[:n_train].copy()
    x_test = all_data.iloc[n_train:].copy()

    x_train.to_csv(PROCESSED_DIR / "train_processed_X.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "train_processed_y.csv", index=False, header=True)
    x_test.to_csv(PROCESSED_DIR / "test_processed_X.csv", index=False)

    summary = "\n".join(
        [
            "# On Isleme Ozeti",
            "",
            f"Ham egitim boyutu: {original_train_rows} satir",
            f"Ham test boyutu: {test.shape[0]} satir",
            f"Silinen aykiri satir: {removed_rows}",
            f"Model egitim boyutu: {x_train.shape[0]} satir x {x_train.shape[1]} ozellik",
            f"Model test boyutu: {x_test.shape[0]} satir x {x_test.shape[1]} ozellik",
            f"Log donusumu uygulanan carpik ozellik sayisi: {len(high_skew)}",
            "",
        ]
    )
    (RESULTS_DIR / "preprocessing_summary.md").write_text(summary, encoding="utf-8")

    print("On isleme tamamlandi.")
    print(f"Egitim matrisi: {x_train.shape}")
    print(f"Test matrisi: {x_test.shape}")
    print(f"Ozet: {(RESULTS_DIR / 'preprocessing_summary.md').relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
