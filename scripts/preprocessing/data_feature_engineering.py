# scripts/preprocessing/data_feature_engineering.py

import sys
import pathlib
# Proje kökünü path'e ekle
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from config import PROCESSED_DIR


def main():
    # 1) Ön işlenmiş veriyi oku
    input_path = PROCESSED_DIR / "student_habits_preprocessed.csv"
    df = pd.read_csv(input_path)

    # 2) Etkileşim ve polinom özellikleri ekle
    df['study_attendance_interaction'] = df['study_hours_per_day'] * df['attendance_percentage']
    df['total_social_hours']            = df['social_media_hours'] + df['netflix_hours']
    df['social_to_study_ratio']         = df['total_social_hours'] / (df['study_hours_per_day'] + 1)
    df['sleep_study_ratio']             = df['sleep_hours'] / (df['study_hours_per_day'] + 1)

    # 3) FE sonrası veriyi kaydet
    output_path = PROCESSED_DIR / "student_habits_fe.csv"
    df.to_csv(output_path, index=False)

    print(f"Özellik mühendisliği tamamlandı. Kaydedilen dosya: {output_path}")
    print("\nÖrnek satırlar:\n", df.head())

if __name__ == "__main__":
    main()
