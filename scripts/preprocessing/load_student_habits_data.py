# scripts/preprocessing/load_student_habits_data.py

import sys
import pathlib
# Proje kökünü path'e ekle
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from config import RAW_DIR, ORIGINAL_DIR


def main():
    # Kaggle API ile veri setini indir ve aç
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        'jayaantanaath/student-habits-vs-academic-performance',
        path=str(RAW_DIR),
        unzip=True
    )

    # İndirilen dosyaları listele
    print("RAW_DIR içeriği:", os.listdir(RAW_DIR))

    # Çıkan orijinal CSV'i ORIGINAL_DIR altına taşı (isteğe bağlı)
    # os.replace(RAW_DIR / 'student_habits_performance.csv', ORIGINAL_DIR / 'student_habits_performance.csv')

    # CSV'i oku ve ilk satırları göster
    df = pd.read_csv(ORIGINAL_DIR / "student_habits_performance.csv")
    print(df.head())

if __name__ == "__main__":
    main()
