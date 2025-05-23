# Student Habits vs Academic Performance Prediction

Bu proje, Kaggle üzerindeki [Student Habits vs Academic Performance](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance) veri seti kullanılarak, yapay sinir ağları (ANN) ile öğrencilerin sınav skorlarını tahmin etmeyi amaçlamaktadır.

## 📂 Proje Yapısı

```
ANN_term_project/
├── .gitignore
├── README.md
├── config.py
├── data/                   # Ham ve işlenmiş veriler
│   ├── raw/                # Kaggle'dan indirilen ham CSV
│   ├── processed/          # Ön işleme, FE ve ölçekleme sonrası CSV'ler
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   ├── test.csv        # Demo için tek satırlık örnek
│   │   ├── student_habits_fe.csv
│   │   └── student_habits_scaled.csv
├── demo/
│   └── demo_predict.py     # İşlenmiş test.csv ile tek örnek tahmin
├── models/                 # Kaydedilmiş ANN modelleri ve history
│   ├── baseline_ann.h5
│   ├── fe_baseline_ann.h5
│   ├── tuned_ann.h5
│   ├── fe_tuned_ann.h5
│   ├── history.pkl
│   └── fe_tuned_history.pkl
├── requirements.txt        # Python paket bağımlılıkları
├── scripts/
│   ├── preprocessing/      # Veri indirme, ön işleme, FE, ölçekleme, bölme
│   │   ├── load_student_habits_data.py
│   │   ├── data_preprocessing.py
│   │   ├── data_feature_engineering.py
│   │   ├── data_scaling.py
│   │   └── data_split.py
│   ├── eda/                # Keşifsel veri analizi ve grafikler
│   │   ├── eda.py
│   │   ├── eda_correlation.py
│   │   ├── eda_plots.py
│   │   ├── plot_history.py
│   │   └── plot_tuned_history.py
│   └── modeling/           # Model eğitimi ve cross-validation
│       ├── train_baseline_ann.py
│       ├── train_tuned_ann.py
│       ├── tune_ann.py
│       ├── tune_ann_hyperband.py
│       └── cross_val_fe_tuned.py
└── test.csv                # (İsteğe bağlı) Ham demo testi için tek satırlık CSV
```

## ⚙️ Kurulum & Çalıştırma



### Örnek `requirements.txt`

```text
pandas==1.5.3
numpy==1.24.2
scikit-learn==1.2.2
tensorflow==2.11.0
keras-tuner==1.1.3
matplotlib==3.7.1
kaggle==1.5.17
```

*(Sürüm numaralarını kendi ortamınıza göre güncelleyebilirsiniz.)*

1. **Clonelayın**

   ```bash
   git clone https://github.com/cagataydisli/ann.git
   cd ANN_term_project
   ```

2. **Sanal ortam oluşturun**

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. **Gerekli paketleri yükleyin**

   ```bash
   pip install -r requirements.txt
   ```

4. **Veriyi indirin ve ön işleyin**

   ```bash
   python scripts/preprocessing/load_student_habits_data.py
   python scripts/preprocessing/data_preprocessing.py
   python scripts/preprocessing/data_feature_engineering.py
   python scripts/preprocessing/data_scaling.py
   python scripts/preprocessing/data_split.py
   ```

5. **Keşifsel Veri Analizi (EDA)**

   ```bash
   python scripts/eda/eda.py
   python scripts/eda/eda_correlation.py
   python scripts/eda/eda_plots.py
   python scripts/eda/plot_history.py
   python scripts/eda/plot_tuned_history.py
   ```

6. **Model Eğitimi**

   * Baseline model:

     ```bash
     python scripts/modeling/train_baseline_ann.py
     ```
   * Tuned model (manuel):

     ```bash
     python scripts/modeling/train_tuned_ann.py
     ```
   * Hyperparameter tuning (Keras Tuner):

     ```bash
     python scripts/modeling/tune_ann.py          # RandomSearch
     python scripts/modeling/tune_ann_hyperband.py # Hyperband
     ```
   * Cross-validation:

     ```bash
     python scripts/modeling/cross_val_fe_tuned.py
     ```

7. **Demo Tahmin**

   * İşlenmiş `data/processed/test.csv` kullanarak single-sample tahmin:

     ```bash
     python demo/demo_predict.py
     ```

## 📄 Lisans

Apache-2.0

---

*Hazırlayan: Çağatay Dişli*
