from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "blastchar/telco-customer-churn"
OUTPUT_DIR = Path("data")


def cli_available() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False


def credentials_available() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_json = kaggle_json.exists()
    has_env = bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))
    return has_json or has_env


def install_cli() -> None:
    print("Installation de l’API Kaggle...")
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)


def download_dataset() -> int:
    if not cli_available():
        try:
            install_cli()
        except subprocess.CalledProcessError:
            print("Échec de l’installation. Vérifie que Python/pip est bien disponible.")
            return 1

    if not credentials_available():
        print("Aucun identifiant Kaggle détecté.")
        print("1. Crée ton token sur https://www.kaggle.com/settings")
        print("2. Télécharge kaggle.json")
        print("3. Place-le dans :")
        print(f"   {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("4. Puis relance ce script.")
        return 1

    OUTPUT_DIR.mkdir(exist_ok=True)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(OUTPUT_DIR),
        "--unzip",
    ]

    print(f"Téléchargement du dataset {DATASET_SLUG} dans {OUTPUT_DIR}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Le téléchargement a échoué : {exc}")
        return 1

    print("Téléchargement terminé.")
    print(f"Fichier CSV attendu : {OUTPUT_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(download_dataset())
