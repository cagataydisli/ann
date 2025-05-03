# scripts/preprocessing/data_scaling.py

import sys
import pathlib
# Proje kökünü path'e ekle
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import PROCESSED_DIR


def main():
    # 1) FE sonrası veriyi oku
    fe_path = PROCESSED_DIR / "student_habits_fe.csv"
    df = pd.read_csv(fe_path)

    # 2) Hedef değişkeni ayır
    y = df["exam_score"]
    X = df.drop(columns=["student_id", "exam_score"])

    # 3) Ölçeklenecek sayısal sütunları belirle
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # 4) StandardScaler ile dönüştür
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # 5) Ölçeklenmiş veriyi birleştir ve kaydet
    df_scaled = pd.concat([X, y], axis=1)
    scaled_path = PROCESSED_DIR / "student_habits_scaled.csv"
    df_scaled.to_csv(scaled_path, index=False)

    print(f"Özellik ölçeklendirme tamamlandı. Kaydedilen dosya: {scaled_path}")

if __name__ == "__main__":
    main()
