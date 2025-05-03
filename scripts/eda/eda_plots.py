import pandas as pd
import matplotlib.pyplot as plt
from config import ORIGINAL_DIR


def main():
    # 1) Veri setini oku
    raw_path = ORIGINAL_DIR / "student_habits_performance.csv"
    df = pd.read_csv(raw_path)

    # 2) Sayısal sütunları seç
    numeric_cols = [
        "age", "study_hours_per_day", "social_media_hours", "netflix_hours",
        "attendance_percentage", "sleep_hours", "exercise_frequency",
        "mental_health_rating", "exam_score"
    ]

    # 3) Her sayısal sütun için histogram
    for col in numeric_cols:
        plt.figure()
        df[col].hist(bins=20)
        plt.title(f"{col} Dağılımı")
        plt.xlabel(col)
        plt.ylabel("Frekans")
        plt.tight_layout()
        plt.show()

    # 4) Kategorik sütunlar için bar grafikleri
    cat_cols = [
        "gender", "part_time_job", "diet_quality",
        "parental_education_level", "internet_quality", "extracurricular_participation"
    ]

    for col in cat_cols:
        plt.figure()
        df[col].value_counts(dropna=False).plot(kind="bar")
        plt.title(f"{col} Frekans Dağılımı")
        plt.xlabel(col)
        plt.ylabel("Adet")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    # 5) Korelasyon matrisi ve ısı haritası
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    plt.matshow(corr, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title("Sayısal Değişkenler Korelasyon Matrisi", pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
