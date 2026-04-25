from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    datasets_dir=Path(r"D:\code\ml-learning-journey\chapter02\datasets")
    tarball_path = datasets_dir / "housing.tgz"
    if not tarball_path.is_file():
        datasets_dir.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=datasets_dir, filter="data")
    return pd.read_csv(datasets_dir / "housing" / "housing.csv")

housing_full = load_housing_data()