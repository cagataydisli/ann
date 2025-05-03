import pandas as pd
from config import ORIGINAL_DIR


def main():
    # 1) Veri setini yükle
    raw_path = ORIGINAL_DIR / "student_habits_performance.csv"
    df = pd.read_csv(raw_path)

    # 2) Sütun tipleri ve doluluk durumu
    print("=== Sütun Tipleri ve Non-Null Sayısı ===")
    df.info()
    print()

    # 3) Sayısal sütunların temel istatistikleri
    print("=== Sayısal Değişkenler için Describe() ===")
    print(df.describe(), "\n")

    # 4) Eksik değer analizi
    missing = df.isnull().sum().to_frame("missing_count")
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
    print("=== Eksik Değerler ===")
    print(missing)

if __name__ == "__main__":
    main()