import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
import pandas as pd
from config import ORIGINAL_DIR, PROCESSED_DIR


def main():
    # 1) Ham veriyi oku
    raw_path = ORIGINAL_DIR / "student_habits_performance.csv"
    df = pd.read_csv(raw_path)

    # 2) Eksik değerleri doldur
    df["parental_education_level"] = df["parental_education_level"].fillna("Unknown")

    # 3) Ordinal değişkenler için mapping
    diet_map     = {"Poor": 0, "Fair": 1, "Good": 2}
    internet_map = {"Poor": 0, "Average": 1, "Good": 2}
    edu_map      = {"Unknown": 0, "High School": 1, "Bachelor": 2, "Master": 3}

    df["diet_quality_ord"]              = df["diet_quality"].map(diet_map)
    df["internet_quality_ord"]         = df["internet_quality"].map(internet_map)
    df["parental_education_level_ord"] = df["parental_education_level"].map(edu_map)

    # 4) Nominal kategorikler için one-hot encoding
    df = pd.get_dummies(
        df,
        columns=["gender", "part_time_job", "extracurricular_participation"],
        drop_first=True
    )

    # 5) Orijinal kategorik sütunları at
    df = df.drop(columns=["diet_quality",
                          "internet_quality",
                          "parental_education_level"])

    # 6) Ön işlenmiş veriyi kaydet
    preprocessed_path = PROCESSED_DIR / "student_habits_preprocessed.csv"
    df.to_csv(preprocessed_path, index=False)

    print(f"Ön işleme tamamlandı. Kaydedilen dosya: {preprocessed_path}")


if __name__ == "__main__":
    main()
