import pandas as pd
from tensorflow import keras
import keras_tuner as kt
from config import PROCESSED_DIR, MODELS_DIR


def main():
    # 1) Veriyi yükle
    train_path = PROCESSED_DIR / "train.csv"
    val_path   = PROCESSED_DIR / "validation.csv"
    test_path  = PROCESSED_DIR / "test.csv"

    train = pd.read_csv(train_path).apply(pd.to_numeric, errors="coerce")
    val   = pd.read_csv(val_path).apply(pd.to_numeric, errors="coerce")
    test  = pd.read_csv(test_path).apply(pd.to_numeric, errors="coerce")

    X_train = train.drop(columns=["exam_score"]).values.astype("float32")
    y_train = train["exam_score"].values.astype("float32")
    X_val   = val.drop(columns=["exam_score"]).values.astype("float32")
    y_val   = val["exam_score"].values.astype("float32")
    X_test  = test.drop(columns=["exam_score"]).values.astype("float32")
    y_test  = test["exam_score"].values.astype("float32")

    n_features = X_train.shape[1]

    # 2) Model kurucusu (HyperModel)
    def build_model(hp):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(n_features,)))
        # 2–4 gizli katman
        for i in range(hp.Int("num_layers", 2, 4)):
            units = hp.Int(f"units_{i}", 32, 256, step=32)
            model.add(keras.layers.Dense(units, activation="relu"))
            dropout = hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)
            if dropout > 0:
                model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1))  # Regresyon çıkışı

        lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mean_squared_error",
            metrics=["mean_absolute_error"]
        )
        return model

    # 3) Random Search Tuner
    tuner = kt.RandomSearch(
        build_model,
        objective="val_mean_absolute_error",
        max_trials=20,
        executions_per_trial=1,
        directory="kt_tuner",
        project_name="student_habits"
    )

    # 4) Arama
    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]
    )

    # 5) En iyi hiperparametreler
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("🏆 En iyi hiperparametreler:")
    for key, val in best_hp.values.items():
        print(f"  {key}: {val}")

    # 6) En iyi modeli kaydet ve test et
    best_model = tuner.get_best_models(1)[0]
    model_path = MODELS_DIR / "kt_tuned_ann.keras"
    best_model.save(model_path)
    loss, mae = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {loss:.4f}, Test MAE: {mae:.4f}")
    print(f"Model kaydedildi: {model_path}")


if __name__ == "__main__":
    main()