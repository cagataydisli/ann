# config.py

from pathlib import Path

# Proje kök klasörü
BASE_DIR = Path(__file__).resolve().parent

# Veri dizinleri
RAW_DIR       = BASE_DIR / "data" / "raw"
ORIGINAL_DIR  = BASE_DIR / "data" / "original"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Modeller için dizin
MODELS_DIR    = BASE_DIR / "models"

# Skript klasörleri (opsiyonel, eğer iç içe import yapacaksan)
SCRIPTS_DIR   = BASE_DIR / "scripts"
