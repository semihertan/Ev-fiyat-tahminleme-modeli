from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
RESULTS_DIR = ROOT_DIR / "outputs" / "results"


def evaluate_model(model, x_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
    predictions_log = model.predict(x_val)
    predictions = np.expm1(predictions_log)
    actual = np.expm1(y_val)

    return {
        "RMSE ($)": float(np.sqrt(mean_squared_error(actual, predictions))),
        "RMSLE": float(np.sqrt(mean_squared_error(y_val, predictions_log))),
        "R2 Score": float(r2_score(y_val, predictions_log)),
    }


def save_model_comparison(results: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    ordered = results.sort_values("RMSE ($)")
    sns.barplot(x="RMSE ($)", y="Model", data=ordered, palette="viridis")
    plt.title("Model Hata Oranlari Karsilastirmasi")
    plt.xlabel("Ortalama Hata Payi ($)")
    plt.ylabel("Model")

    for index, rmse in enumerate(ordered["RMSE ($)"]):
        plt.text(rmse + 500, index, f"${rmse:,.0f}", va="center", color="black")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sonuc_4_model_karsilastirma.png", dpi=150)
    plt.close()


def save_prediction_plot(model, model_name: str, x_val: pd.DataFrame, y_val: pd.Series) -> None:
    predictions = np.expm1(model.predict(x_val))
    actual = np.expm1(y_val)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=actual, y=predictions, alpha=0.6, color="purple")
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--", linewidth=2)
    plt.title(f"En Iyi Model ({model_name}): Gercek vs Tahmin")
    plt.xlabel("Gercek Fiyatlar ($)")
    plt.ylabel("Tahmin Edilen Fiyatlar ($)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sonuc_en_iyi_tahmin.png", dpi=150)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x = pd.read_csv(PROCESSED_DIR / "train_processed_X.csv")
    y = pd.read_csv(PROCESSED_DIR / "train_processed_y.csv").iloc[:, 0]
    x_test = pd.read_csv(PROCESSED_DIR / "test_processed_X.csv")

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
    )

    models = {
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=0.0005, random_state=42, max_iter=10000),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    rows = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"Egitiliyor: {model_name}")
        fitted_model = clone(model)
        fitted_model.fit(x_train, y_train)
        fitted_models[model_name] = fitted_model

        metrics = evaluate_model(fitted_model, x_val, y_val)
        rows.append({"Model": model_name, **metrics})

    results = pd.DataFrame(rows).sort_values("RMSE ($)").reset_index(drop=True)
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    best_model_name = results.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    save_model_comparison(results)
    save_prediction_plot(best_model, best_model_name, x_val, y_val)

    final_model = clone(models[best_model_name])
    final_model.fit(x, y)

    test_ids = pd.read_csv(RAW_DIR / "test.csv")["Id"]
    test_predictions = np.expm1(final_model.predict(x_test))
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_predictions})
    submission.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    best = results.iloc[0]
    summary = "\n".join(
        [
            "# Model Sonuc Ozeti",
            "",
            f"En iyi model: {best['Model']}",
            f"RMSE ($): {best['RMSE ($)']:,.2f}",
            f"RMSLE: {best['RMSLE']:.4f}",
            f"R2 Score: {best['R2 Score']:.4f}",
            "",
            "Tum model sonuclari `model_results.csv` dosyasinda saklanir.",
            "Test seti tahminleri `test_predictions.csv` dosyasina yazilir.",
            "",
        ]
    )
    (RESULTS_DIR / "best_model_summary.md").write_text(summary, encoding="utf-8")

    print("\nModel karsilastirma sonuclari:")
    print(results.to_string(index=False))
    print(f"\nEn iyi model: {best_model_name}")
    print(f"Sonuclar: {RESULTS_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
