# scripts/eda/plot_history.py

import pickle
import matplotlib.pyplot as plt
from config import MODELS_DIR


def main():
    # 1) Eğitim sırasında kaydedilen history objesini yükle
    history_path = MODELS_DIR / "baseline_history.pkl"
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    # 2) MSE Loss grafiği
    plt.figure()
    plt.plot(history["loss"], label="Eğitim Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Epoch’a Göre MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) MAE grafiği
    plt.figure()
    plt.plot(history["mean_absolute_error"], label="Eğitim MAE")
    plt.plot(history["val_mean_absolute_error"], label="Validation MAE")
    plt.title("Epoch’a Göre MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
