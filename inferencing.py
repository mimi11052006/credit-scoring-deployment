"""
inferencing.py
===============
Code inferencing untuk model Credit Score Classification.

Berisi fungsi-fungsi untuk:
- memuat model + preprocessor hasil training (pipeline.py)
- menerima satu data nasabah (raw, sesuai kolom asli sebelum preprocessing)
- menjalankan preprocessing yang identik dengan training (lewat class Preprocessing)
- mengembalikan hasil prediksi Credit_Score beserta probabilitas tiap kelas

Catatan penting:
- Karena Preprocessing adalah class custom (didefinisikan di pipeline.py),
  modul ini WAJIB mengimpor `pipeline` terlebih dahulu sebelum joblib.load(),
  supaya Python mengenali tipe objek yang di-pickle.
- File ini didesain untuk dipanggil dari Streamlit (app.py), tapi juga bisa
  dijalankan mandiri lewat command line untuk uji cepat (lihat bagian __main__).
"""

import os
import joblib
import numpy as np
import pandas as pd

# WAJIB: import dulu modul pipeline supaya class Preprocessing dikenali saat unpickle
import pipeline  # noqa: F401
from pipeline import Preprocessing  # noqa: F401


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "preprocessor.pkl")


def load_artifacts(model_path: str = MODEL_PATH, preprocessor_path: str = PREPROCESSOR_PATH):
    """Memuat model dan preprocessor yang sudah di-fit dari hasil training (pipeline.py)."""
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def predict_single(data: dict, model, preprocessor) -> dict:
    """
    Melakukan prediksi untuk satu data nasabah.

    Parameter
    ---------
    data : dict
        Data nasabah dalam bentuk dictionary, dengan key sesuai nama kolom
        mentah pada dataset asli (sebelum preprocessing). Lihat FIELD_SPEC
        di app.py untuk daftar lengkap field yang dibutuhkan.
    model : object
        Pipeline model (SMOTE + estimator) hasil training, dimuat dari model.pkl.
    preprocessor : Preprocessing
        Objek Preprocessing yang sudah di-fit, dimuat dari preprocessor.pkl.

    Return
    ------
    dict berisi:
        - 'predicted_class'  : label kelas hasil prediksi (Good/Standard/Poor)
        - 'probabilities'    : dict {nama_kelas: probabilitas}
    """
    df_input = pd.DataFrame([data])

    # transform() mengembalikan (X, y) -- y akan None karena tidak ada kolom Credit_Score
    X_transformed, _ = preprocessor.transform(df_input)

    pred_encoded = model.predict(X_transformed)[0]
    pred_label = preprocessor.le_target_.inverse_transform([pred_encoded])[0]

    proba_result = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_transformed)[0]
        for kelas, p in zip(preprocessor.le_target_.classes_, proba):
            proba_result[kelas] = float(p)

    return {
        "predicted_class": pred_label,
        "probabilities": proba_result
    }


def predict_batch(df_input: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    """
    Melakukan prediksi untuk banyak baris sekaligus (mis. dari file CSV upload).
    Mengembalikan DataFrame asli + kolom tambahan 'Predicted_Credit_Score'.
    """
    X_transformed, _ = preprocessor.transform(df_input)
    pred_encoded = model.predict(X_transformed)
    pred_label = preprocessor.le_target_.inverse_transform(pred_encoded)

    df_result = df_input.copy()
    df_result["Predicted_Credit_Score"] = pred_label
    return df_result


if __name__ == "__main__":
    # Contoh pemakaian mandiri (tanpa Streamlit), untuk uji cepat dari command line.
    model, preprocessor = load_artifacts()

    contoh_data = {
        "Age": 32,
        "Occupation": "Doctor",
        "Annual_Income": 56125.5,
        "Monthly_Inhand_Salary": 4875.13,
        "Num_Bank_Accounts": 8,
        "Num_Credit_Card": 3,
        "Interest_Rate": 18,
        "Num_of_Loan": 2,
        "Type_of_Loan": "Credit-Builder Loan, and Mortgage Loan",
        "Delay_from_due_date": 30,
        "Num_of_Delayed_Payment": 14,
        "Changed_Credit_Limit": 17.89,
        "Num_Credit_Inquiries": 4.0,
        "Credit_Mix": "Standard",
        "Outstanding_Debt": 370.22,
        "Credit_Utilization_Ratio": 32.01,
        "Credit_History_Age": "28 Years and 10 Months",
        "Payment_of_Min_Amount": "Yes",
        "Total_EMI_per_month": 81.82,
        "Amount_invested_monthly": 182.07,
        "Payment_Behaviour": "High_spent_Medium_value_payments",
        "Monthly_Balance": 473.62,
    }

    hasil = predict_single(contoh_data, model, preprocessor)
    print("Prediksi Credit Score:", hasil["predicted_class"])
    print("Probabilitas tiap kelas:")
    for kelas, p in hasil["probabilities"].items():
        print(f"  {kelas}: {p:.4f}")
