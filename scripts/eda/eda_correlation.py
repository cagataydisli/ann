import pandas as pd
import matplotlib.pyplot as plt
from config import ORIGINAL_DIR

# 1) Veri setini oku
raw_path = ORIGINAL_DIR / "student_habits_performance.csv"
df = pd.read_csv(raw_path)

# 2) Sayısal sütunları seç
numeric = [
    "age", "study_hours_per_day", "social_media_hours", "netflix_hours",
    "attendance_percentage", "sleep_hours", "exercise_frequency",
    "mental_health_rating", "exam_score"
]

# 3) Korelasyon matrisini oluştur
corr = df[numeric].corr()

# 4) Isı haritası çiz
plt.figure(figsize=(8,6))
plt.matshow(corr, fignum=1)
plt.colorbar()
plt.xticks(range(len(numeric)), numeric, rotation=90)
plt.yticks(range(len(numeric)), numeric)
plt.title("Sayısal Değişkenler Korelasyon Matrisi", pad=20)
plt.tight_layout()
plt.show()
