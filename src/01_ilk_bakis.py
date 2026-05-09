from pathlib import Path
import io

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
RESULTS_DIR = ROOT_DIR / "outputs" / "results"
TRAIN_PATH = RAW_DIR / "train.csv"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_PATH)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)

    buffer = io.StringIO()
    df.info(buf=buffer)

    report = [
        "# Veri Ilk Bakis Raporu",
        "",
        f"Kaynak dosya: {TRAIN_PATH.relative_to(ROOT_DIR)}",
        f"Satir sayisi: {df.shape[0]}",
        f"Sutun sayisi: {df.shape[1]}",
        "",
        "## Ilk 5 Satir",
        df.head().to_string(),
        "",
        "## Veri Tipleri ve Eksik Degerler",
        buffer.getvalue(),
        "",
        "## Istatistiksel Ozet",
        df.describe().T.to_string(),
        "",
    ]

    output_path = RESULTS_DIR / "data_overview.txt"
    output_path.write_text("\n".join(report), encoding="utf-8")

    print("Veri basariyla incelendi.")
    print(f"Rapor kaydedildi: {output_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
