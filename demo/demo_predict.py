# demo/demo_predict.py

import sys
import pathlib
# Proje kökünü path'e ekle ki config modülünü import edebilelim
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from config import PROCESSED_DIR, MODELS_DIR


def main():
    # 1) İşlenmiş test setini oku (tek satırlık örnek)
    test_path = PROCESSED_DIR / "test.csv"
    df_test = pd.read_csv(test_path).apply(pd.to_numeric, errors="coerce")

        # 2) Özellikleri al (student_id ve exam_score sütunlarını kaldır)
    X_df = df_test.drop(columns=["student_id", "exam_score"], errors="ignore")

    # 3) Modelin beklediği kolon sırasını al ve hizala
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    feature_cols = [c for c in train_df.columns if c not in ["student_id", "exam_score"]]
    X_df = X_df.reindex(columns=feature_cols, fill_value=0)
    X_sample = np.expand_dims(X_df.iloc[0].values.astype("float32"), axis=0)

    # 4) Eğitilmiş modeli yükle
    model = load_model(MODELS_DIR / "fe_tuned_ann.h5")

    # 4) Tahmin yap
    pred = model.predict(X_sample)

    # 5) Çıktı
    if "exam_score" in df_test.columns:
        print(f"Gerçek sınav skoru: {df_test['exam_score'].iloc[0]}")
    print(f"Tahmin edilen sınav skoru: {pred[0,0]:.2f}")

if __name__ == "__main__":
    main()
