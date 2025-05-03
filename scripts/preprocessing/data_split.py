# scripts/preprocessing/data_split.py

import sys
import pathlib
# Proje kökünü path'e ekle
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from config import PROCESSED_DIR


def main():
    # 1) Ölçeklenmiş veriyi oku
    scaled_path = PROCESSED_DIR / "student_habits_scaled.csv"
    df = pd.read_csv(scaled_path)

    # 2) Özellikleri ve hedefi ayır (student_id ve exam_score düşülüyor)
    X = df.drop(columns=["student_id", "exam_score"], errors="ignore")
    y = df["exam_score"]

    # 3) Önce test setini ayır (%15)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # 4) Kalan %85’ten validation setini ayır (~17.65%)
    val_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42
    )

    # 5) DataFrame’leri birleştir ve kaydet
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df   = pd.concat([X_val,   y_val],   axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)

    train_path = PROCESSED_DIR / "train.csv"
    val_path   = PROCESSED_DIR / "validation.csv"
    test_path  = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 6) Set boyutlarını yazdır
    print(f"Eğitim seti: {train_df.shape}")
    print(f"Validation seti: {val_df.shape}")
    print(f"Test seti: {test_df.shape}")

if __name__ == "__main__":
    main()
