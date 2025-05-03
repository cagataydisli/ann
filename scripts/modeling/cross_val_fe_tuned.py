# scripts/modeling/cross_val_fe_tuned.py

import sys
import pathlib
# Proje kökünü path'e ekle
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model
from config import PROCESSED_DIR, MODELS_DIR


def main():
    # 1) Ölçeklenmiş, FE uygulanmış tam veri setini oku
    scaled_path = PROCESSED_DIR / "student_habits_scaled.csv"
    df = pd.read_csv(scaled_path)

    # 2) Özellikleri ve hedefi ayır (student_id, exam_score sütunları yoksa ignore)
    X = df.drop(columns=["student_id", "exam_score"], errors="ignore").values.astype("float32")
    y = df["exam_score"].values.astype("float32")

    # 3) 5-Fold CV ayarları
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    # 4) Her fold’da değerlendirme
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_val, y_val = X[val_idx], y[val_idx]

        # Eğitilmiş FE+tuned modelini her fold için yükle
        model = load_model(MODELS_DIR / "fe_tuned_ann.h5")

        # Değerlendirme
        loss, mae = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold} → MSE: {loss:.2f}, MAE: {mae:.2f}")
        scores.append((loss, mae))

    # 5) Ortalama ve standart sapma
    mse_scores, mae_scores = zip(*scores)
    print(f"\n5-Fold CV Ortalama MSE: {np.mean(mse_scores):.2f} ± {np.std(mse_scores):.2f}")
    print(f"5-Fold CV Ortalama MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")

if __name__ == "__main__":
    main()
