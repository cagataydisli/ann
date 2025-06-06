import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from config import PROCESSED_DIR, MODELS_DIR
import pickle


def main():
    # 1) Bölünmüş veri setlerini oku
    train_path = PROCESSED_DIR / "train.csv"
    val_path   = PROCESSED_DIR / "validation.csv"
    test_path  = PROCESSED_DIR / "test.csv"

    train = pd.read_csv(train_path).apply(pd.to_numeric, errors="coerce")
    val   = pd.read_csv(val_path).apply(pd.to_numeric, errors="coerce")
    test  = pd.read_csv(test_path).apply(pd.to_numeric, errors="coerce")

    # 2) Özellikleri ve hedefi ayır
    X_train = train.drop(columns=["student_id","exam_score"]).values.astype("float32")
    y_train = train["exam_score"].values.astype("float32")
    X_val   = val.drop(columns=["student_id","exam_score"]).values.astype("float32")
    y_val   = val["exam_score"].values.astype("float32")
    X_test  = test.drop(columns=["student_id","exam_score"]).values.astype("float32")
    y_test  = test["exam_score"].values.astype("float32")

    # 3) Model mimarisi
    n_features = X_train.shape[1]
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1)
    ])

    # 4) Derleme
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )

    # 5) Early stopping
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # 6) Eğitim
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=2
    )

    # 7) Değerlendir
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {loss:.4f}, MAE: {mae:.4f}")

    # 8) Modeli kaydet
    model_path = MODELS_DIR / "baseline_ann.h5"
    model.save(model_path)
    print(f"Model kaydedildi: {model_path}")

    # 9) Eğitim geçmişini kaydet
    history_path = MODELS_DIR / "baseline_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"Eğitim geçmişi kaydedildi: {history_path}")


if __name__ == "__main__":
    main()