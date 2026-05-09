from pathlib import Path
import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
RESULTS_DIR = ROOT_DIR / "outputs" / "results"
TRAIN_PATH = RAW_DIR / "train.csv"


def save_current_figure(file_name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=150)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rc("figure", figsize=(12, 8))
    plt.rc("font", size=11)
    plt.rc("axes", titlesize=15, labelsize=12)

    df = pd.read_csv(TRAIN_PATH)

    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)

    buffer = io.StringIO()
    df.info(buf=buffer)

    (RESULTS_DIR / "eda_dataset_info.txt").write_text(buffer.getvalue(), encoding="utf-8")

    plt.figure(figsize=(12, 6))
    sns.histplot(df["SalePrice"], kde=True, bins=50)
    plt.title("SalePrice Dagilimi")
    plt.xlabel("Satis Fiyati ($)")
    plt.ylabel("Frekans")
    save_current_figure("01_saleprice_dagilimi.png")

    df["SalePrice_log"] = np.log1p(df["SalePrice"])
    plt.figure(figsize=(12, 6))
    sns.histplot(df["SalePrice_log"], kde=True, bins=50, color="seagreen")
    plt.title("Log Donusumlu SalePrice Dagilimi")
    plt.xlabel("Log(Satis Fiyati)")
    plt.ylabel("Frekans")
    save_current_figure("02_saleprice_log_dagilimi.png")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["GrLivArea"], y=df["SalePrice"], alpha=0.75)
    plt.title("Yasam Alani (GrLivArea) ve Satis Fiyati")
    plt.xlabel("Yasam Alani (sqft)")
    plt.ylabel("Satis Fiyati ($)")
    save_current_figure("03_yasamalani_vs_fiyat.png")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["TotalBsmtSF"], y=df["SalePrice"], alpha=0.75)
    plt.title("Toplam Bodrum Alani ve Satis Fiyati")
    plt.xlabel("Toplam Bodrum Alani (sqft)")
    plt.ylabel("Satis Fiyati ($)")
    save_current_figure("04_bodrumalani_vs_fiyat.png")

    plt.figure(figsize=(12, 7))
    sns.boxplot(x=df["OverallQual"], y=df["SalePrice"])
    plt.title("Genel Kalite (OverallQual) ve Satis Fiyati")
    plt.xlabel("Genel Kalite")
    plt.ylabel("Satis Fiyati ($)")
    save_current_figure("05_genelkalite_vs_fiyat.png")

    quality_order = ["Po", "Fa", "TA", "Gd", "Ex"]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df["KitchenQual"], y=df["SalePrice"], order=quality_order)
    plt.title("Mutfak Kalitesi ve Satis Fiyati")
    plt.xlabel("Mutfak Kalitesi")
    plt.ylabel("Satis Fiyati ($)")
    save_current_figure("06_mutfakkalitesi_vs_fiyat.png")

    plt.figure(figsize=(20, 10))
    neighborhood_order = df.groupby("Neighborhood")["SalePrice"].median().sort_values().index
    sns.boxplot(x=df["Neighborhood"], y=df["SalePrice"], order=neighborhood_order)
    plt.title("Semtlere Gore Satis Fiyatlari")
    plt.xlabel("Semt")
    plt.ylabel("Satis Fiyati ($)")
    plt.xticks(rotation=90)
    save_current_figure("07_semtler_vs_fiyat.png")

    df_numeric = df.select_dtypes(include=[np.number]).drop(
        columns=["Id", "SalePrice_log"],
        errors="ignore",
    )
    corr_matrix = df_numeric.corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Sayisal Ozellikler Korelasyon Haritasi")
    save_current_figure("08_korelasyon_haritasi.png")

    sale_price_corr = corr_matrix["SalePrice"].sort_values(ascending=False)
    plt.figure(figsize=(10, 12))
    sns.barplot(x=sale_price_corr.values, y=sale_price_corr.index, palette="viridis")
    plt.title("SalePrice ile Korelasyonlar")
    plt.xlabel("Korelasyon Katsayisi")
    plt.ylabel("Ozellik")
    save_current_figure("09_fiyat_korelasyonlari_barplot.png")

    top_corr = sale_price_corr.drop("SalePrice").abs().sort_values(ascending=False).head(12)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=top_corr.values, y=top_corr.index, palette="mako")
    plt.title("SalePrice ile En Guclu Iliskili Ozellikler")
    plt.xlabel("Mutlak Korelasyon")
    plt.ylabel("Ozellik")
    save_current_figure("en_onemli_korelasyonlar.png")

    print("EDA grafikleri ve bilgi raporu basariyla olusturuldu.")
    print(f"Grafikler: {FIGURES_DIR.relative_to(ROOT_DIR)}")
    print(f"Rapor: {(RESULTS_DIR / 'eda_dataset_info.txt').relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
