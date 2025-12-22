import os
import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def run_preprocessing(
    raw_data_path: str,
    output_dir: str,
    target_col: str = "income"
):
    """
    Fungsi untuk melakukan preprocessing data secara otomatis
    dan menyimpan hasilnya dalam format siap latih.
    """

    # Buat folder output jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    # Nama kolom dataset Adult Income
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]

    # Load dataset
    df = pd.read_csv(
        raw_data_path,
        header=None,
        names=columns,
        sep=",",
        skipinitialspace=True
    )

    # Ganti "?" jadi NaN
    df = df.replace("?", np.nan)

    # Hapus duplikat
    df = df.drop_duplicates()

    # Pisahkan fitur & target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Tentukan kolom numerik & kategorikal
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Pipeline preprocessing
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Transform data
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Simpan hasil preprocessing
    sp.save_npz(os.path.join(output_dir, "X_train.npz"), X_train_prep)
    sp.save_npz(os.path.join(output_dir, "X_test.npz"), X_test_prep)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(output_dir, "y_test.npy"), y_test.to_numpy())
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    return {
        "X_train_shape": X_train_prep.shape,
        "X_test_shape": X_test_prep.shape,
        "output_dir": output_dir
    }


if __name__ == "__main__":
    RAW_PATH = "adult_income_raw/adult.data"
    OUTPUT_DIR = "preprocessing/adult_income_preprocessing"

    result = run_preprocessing(
        raw_data_path=RAW_PATH,
        output_dir=OUTPUT_DIR,
        target_col="income"
    )

    print("Preprocessing selesai:")
    print(result)
