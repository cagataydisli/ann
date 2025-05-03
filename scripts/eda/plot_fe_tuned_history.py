import pickle
import matplotlib.pyplot as plt
from config import MODELS_DIR


def main():
    # FE + Tuned modelinin eğitim geçmişini yükle
    history_path = MODELS_DIR / "fe_tuned_history.pkl"
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    # MSE Loss grafiği
    plt.figure()
    plt.plot(history["loss"], label="Eğitim MSE")
    plt.plot(history["val_loss"], label="Doğrulama MSE")
    plt.title("FE + Tuned — Epoch’a Göre MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # MAE grafiği
    plt.figure()
    plt.plot(history["mean_absolute_error"], label="Eğitim MAE")
    plt.plot(history["val_mean_absolute_error"], label="Doğrulama MAE")
    plt.title("FE + Tuned — Epoch’a Göre MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
