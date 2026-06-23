"""
pipeline.py
===========
Pipeline training Machine Learning untuk prediksi Credit_Score (Good/Standard/Poor)
berbasis OOP, dengan eksperimen tracking menggunakan MLflow.

Struktur class:
- Preprocessing : membersihkan data mentah, melakukan feature engineering,
                   imputasi missing value, dan encoding. Mengikuti pola fit/transform
                   (fit hanya pada data train, transform diterapkan ke train & test)
                   sehingga tidak terjadi data leakage.
- Training      : membungkus satu algoritma ML ke dalam imbalanced-learn Pipeline
                   (SMOTE + model), menyediakan fungsi cross-validation dan fit/predict.
- Evaluation    : menghitung metrik evaluasi (accuracy, f1-macro, precision, recall,
                   classification report) dengan label yang sudah dipetakan dengan benar.
- ExperimentRunner : orchestrator yang menjalankan beberapa algoritma ML, mencatat
                   setiap eksperimen ke MLflow (parameter, metrik, model), melakukan
                   hyperparameter tuning pada model terbaik, lalu menyimpan model dan
                   seluruh preprocessing artifacts yang dibutuhkan untuk inferencing.

Cara menjalankan:
    python pipeline.py
<<<<<<< HEAD
=======
    $env:MLFLOW_ALLOW_FILE_STORE = "true"
    mlflow ui --backend-store-uri ./mlruns
>>>>>>> 03bbcfb (First commit dengan Git LFS)

Hasil:
    - Eksperimen tercatat di MLflow (folder ./mlruns), bisa dilihat dengan:
          mlflow ui --backend-store-uri ./mlruns
    - Model terbaik tersimpan di ./artifacts/model.pkl
    - Preprocessing artifacts tersimpan di ./artifacts/preprocessing_artifacts.pkl
"""

import re
import os
import json
import joblib
import numpy as np
import pandas as pd

# MLflow 3.x menonaktifkan filesystem tracking backend secara default
# (mode "maintenance"). Kita tetap memakai local file store (./mlruns)
# sesuai kebutuhan proyek ini, jadi opt-out secara eksplisit di sini.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


RANDOM_STATE = 42


# =============================================================================
# CLASS 1: PREPROCESSING
# =============================================================================
class Preprocessing:
    """
    Menangani seluruh proses pembersihan data dan feature engineering.

    Mengikuti pola fit/transform:
      - fit(df_train)         : mempelajari nilai median/modus/kolom HANYA dari data train
      - transform(df)         : menerapkan pembersihan + imputasi + encoding yang konsisten
      - fit_transform(df_train, df_test) : helper untuk split awal

    Semua nilai yang "dipelajari" (median, modus, mapping, daftar kolom final)
    disimpan sebagai atribut object, sehingga bisa di-pickle dan dipakai ulang
    saat inferencing pada data baru.
    """

    KOLOM_HAPUS = ['Unnamed: 0', 'ID', 'Customer_ID', 'Month', 'Name', 'SSN']

    KOLOM_GANTI_NUMERIK = [
        'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly',
        'Monthly_Balance'
    ]

    KOLOM_NUMERIK_MISSING = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_of_Loan',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance',
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate'
    ]

    KOLOM_KATEGORIKAL_MISSING = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount']

    KOLOM_NOMINAL = ['Occupation', 'Payment_of_Min_Amount', 'Spending_Behavior', 'Payment_Size']

    JENIS_PINJAMAN = [
        'auto loan', 'student loan', 'credit-builder loan', 'personal loan',
        'home equity loan', 'mortgage loan', 'payday loan', 'debt consolidation loan'
    ]

    MIX_MAPPING = {'Bad': 0, 'Standard': 1, 'Good': 2}

    def __init__(self):
        self.imputer_values_ = {}
        self.kolom_final_ = None
        self.le_target_ = LabelEncoder()
        self.is_fitted_ = False

    # ---------------------------------------------------------------
    # Tahap 1: cleaning + feature engineering yang deterministik per baris
    # (tidak menghitung statistik agregat dari data, jadi aman dipakai
    #  baik untuk df_train maupun data baru saat inferencing)
    # ---------------------------------------------------------------
    def _clean_dan_feature_engineering(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()

        for kolom in self.KOLOM_GANTI_NUMERIK:
            if kolom in df.columns:
                df[kolom] = pd.to_numeric(df[kolom], errors='coerce')

        kolom_hapus_ada = [k for k in self.KOLOM_HAPUS if k in df.columns]
        if kolom_hapus_ada:
            df = df.drop(columns=kolom_hapus_ada)

        if 'Occupation' in df.columns:
            df['Occupation'] = df['Occupation'].replace('_______', np.nan)
        if 'Credit_Mix' in df.columns:
            df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
        if 'Payment_of_Min_Amount' in df.columns:
            df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan)
        if 'Payment_Behaviour' in df.columns:
            df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)

        def batas(x, lo, hi):
            if pd.isna(x):
                return x
            if x < lo or x > hi:
                return np.nan
            return x

        if 'Age' in df.columns:
            df['Age'] = df['Age'].apply(lambda x: batas(x, 18, 100))
        if 'Num_Bank_Accounts' in df.columns:
            df['Num_Bank_Accounts'] = df['Num_Bank_Accounts'].apply(lambda x: batas(x, 0, 50))
        if 'Num_Credit_Card' in df.columns:
            df['Num_Credit_Card'] = df['Num_Credit_Card'].apply(lambda x: batas(x, 0, 50))
        if 'Interest_Rate' in df.columns:
            df['Interest_Rate'] = df['Interest_Rate'].apply(lambda x: batas(x, 1, 100))
        if 'Num_of_Delayed_Payment' in df.columns:
            df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].apply(lambda x: batas(x, 0, 100))
        if 'Num_Credit_Inquiries' in df.columns:
            df['Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'].apply(lambda x: batas(x, 0, 100))
        if 'Num_of_Loan' in df.columns:
            df['Num_of_Loan'] = df['Num_of_Loan'].apply(lambda x: batas(x, 0, 100))

        # Credit_History_Age -> total bulan
        if 'Credit_History_Age' in df.columns:
            def konversi_ke_bulan(text):
                text = str(text)
                years = re.search(r'(\d+)\s*Years', text)
                months = re.search(r'(\d+)\s*Months', text)
                y = int(years.group(1)) if years else 0
                m = int(months.group(1)) if months else 0
                return (y * 12) + m
            df['Credit_History_Age'] = df['Credit_History_Age'].apply(konversi_ke_bulan)

        # Type_of_Loan -> kolom biner multi-label
        if 'Type_of_Loan' in df.columns:
            df['Type_of_Loan'] = df['Type_of_Loan'].fillna('Not Specified').astype(str).str.lower()
            for pinjaman in self.JENIS_PINJAMAN:
                nama_kolom = 'Loan_' + pinjaman.replace(' ', '_').title()
                df[nama_kolom] = df['Type_of_Loan'].apply(lambda x, p=pinjaman: 1 if p in x else 0)
            df = df.drop(columns=['Type_of_Loan'])

        # Payment_Behaviour -> Spending_Behavior + Payment_Size
        if 'Payment_Behaviour' in df.columns:
            df['Payment_Behaviour'] = df['Payment_Behaviour'].fillna('Unknown').astype(str)
            df['Spending_Behavior'] = df['Payment_Behaviour'].apply(lambda x: 'High' if 'High' in x else 'Low')

            def ekstrak_ukuran(teks):
                if 'Small' in teks:
                    return 'Small'
                elif 'Medium' in teks:
                    return 'Medium'
                elif 'Large' in teks:
                    return 'Large'
                return 'Unknown'

            df['Payment_Size'] = df['Payment_Behaviour'].apply(ekstrak_ukuran)
            df = df.drop(columns=['Payment_Behaviour'])

        return df

    # ---------------------------------------------------------------
    # fit: pelajari median/modus/mapping/kolom HANYA dari data train
    # ---------------------------------------------------------------
    def fit(self, df_train_raw: pd.DataFrame, target_col: str = 'Credit_Score'):
        df = self._clean_dan_feature_engineering(df_train_raw)

        X_train = df.drop(columns=[target_col])
        y_train = df[target_col]

        # pelajari median (numerik) & modus (kategorikal) dari X_train saja
        for kolom in self.KOLOM_NUMERIK_MISSING:
            if kolom in X_train.columns:
                self.imputer_values_[kolom] = X_train[kolom].median()

        for kolom in self.KOLOM_KATEGORIKAL_MISSING:
            if kolom in X_train.columns:
                self.imputer_values_[kolom] = X_train[kolom].mode()[0]

        # fit label encoder target
        self.le_target_.fit(y_train)

        # terapkan imputasi + encoding ke X_train untuk menentukan kolom final
        X_train_transformed = self._apply_imputation_and_encoding(X_train)
        self.kolom_final_ = X_train_transformed.columns.tolist()

        self.is_fitted_ = True
        return self

    def _apply_imputation_and_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # imputasi pakai nilai yang sudah dipelajari saat fit()
        for kolom, nilai in self.imputer_values_.items():
            if kolom in X.columns:
                X[kolom] = X[kolom].fillna(nilai)

        # encode Credit_Mix manual (ordinal)
        if 'Credit_Mix' in X.columns:
            X['Credit_Mix'] = X['Credit_Mix'].map(self.MIX_MAPPING)

        # one-hot encoding kolom nominal
        kolom_nominal_ada = [k for k in self.KOLOM_NOMINAL if k in X.columns]
        X = pd.get_dummies(X, columns=kolom_nominal_ada, drop_first=True)

        return X

    # ---------------------------------------------------------------
    # transform: terapkan ke data apapun (train, test, atau data baru saat inferencing)
    # ---------------------------------------------------------------
    def transform(self, df_raw: pd.DataFrame, target_col: str = 'Credit_Score'):
        if not self.is_fitted_:
            raise RuntimeError("Preprocessing belum di-fit. Panggil .fit() terlebih dahulu.")

        df = self._clean_dan_feature_engineering(df_raw)

        y_encoded = None
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
            y_encoded = self.le_target_.transform(y)
        else:
            X = df

        X_transformed = self._apply_imputation_and_encoding(X)

        # samakan kolom dengan kolom_final_ hasil fit (kolom baru/hilang ditangani)
        X_transformed = X_transformed.reindex(columns=self.kolom_final_, fill_value=0)

        return X_transformed, y_encoded

    def fit_transform_split(self, df_train_raw: pd.DataFrame, df_test_raw: pd.DataFrame,
                             target_col: str = 'Credit_Score'):
        """Helper: fit pada train, lalu transform train & test sekaligus."""
        self.fit(df_train_raw, target_col=target_col)
        X_train, y_train = self.transform(df_train_raw, target_col=target_col)
        X_test, y_test = self.transform(df_test_raw, target_col=target_col)
        return X_train, y_train, X_test, y_test

    @property
    def target_classes(self):
        return self.le_target_.classes_


# =============================================================================
# CLASS 2: TRAINING
# =============================================================================
class Training:
    """
    Membungkus satu algoritma ML ke dalam pipeline (SMOTE + model).
    SMOTE diletakkan di dalam pipeline agar oversampling hanya dilakukan
    pada fold training saat cross-validation, bukan pada keseluruhan data
    sebelum CV (mencegah leakage dari sampel sintetis SMOTE).
    """

    def __init__(self, model_name: str, estimator, random_state: int = RANDOM_STATE):
        self.model_name = model_name
        self.estimator = estimator
        self.random_state = random_state
        self.pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('model', estimator)
        ])
        self.is_fitted_ = False

    def cross_validate(self, X_train, y_train, cv_splits: int = 5, scoring: str = 'f1_macro'):
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        return scores

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError(f"Model '{self.model_name}' belum di-fit.")
        return self.pipeline.predict(X)

    def tune(self, X_train, y_train, param_distributions: dict,
             n_iter: int = 10, cv_splits: int = 3, scoring: str = 'f1_macro'):
        """Hyperparameter tuning dengan RandomizedSearchCV pada pipeline ini."""
        search = RandomizedSearchCV(
            self.pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_splits,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        self.pipeline = search.best_estimator_
        self.is_fitted_ = True
        return search


# =============================================================================
# CLASS 3: EVALUATION
# =============================================================================
class Evaluation:
    """
    Menghitung metrik evaluasi model. target_names selalu diambil dari
    encoder yang sebenarnya (bukan ditulis manual) agar label di laporan
    selalu sesuai urutan hasil encoding yang sebenarnya.
    """

    def __init__(self, target_names):
        self.target_names = list(target_names)

    def compute_metrics(self, y_true, y_pred) -> dict:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        }

    def classification_report_text(self, y_true, y_pred) -> str:
        return classification_report(y_true, y_pred, target_names=self.target_names)

    def confusion_matrix_df(self, y_true, y_pred) -> pd.DataFrame:
        cm = confusion_matrix(y_true, y_pred)
        return pd.DataFrame(cm, index=self.target_names, columns=self.target_names)


# =============================================================================
# ORCHESTRATOR: EXPERIMENT RUNNER
# =============================================================================
class ExperimentRunner:
    """
    Menjalankan eksperimen perbandingan beberapa model ML, mencatat setiap
    eksperimen (parameter, metrik, model) ke MLflow, melakukan hyperparameter
    tuning pada model terbaik, lalu menyimpan model + preprocessing artifacts
    yang dibutuhkan untuk inferencing.
    """

    MODELS = {
        "Decision_Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Extra_Trees": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "Random_Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
    }

    TUNING_PARAM_DIST = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
    }

    def __init__(self, data_path: str, experiment_name: str = "Credit_Score_Classification",
                 artifacts_dir: str = "artifacts", tracking_uri: str = "./mlruns"):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.preprocessor = Preprocessing()
        self.results_ = []
        self.best_model_name_ = None
        self.best_trainer_ = None
        self.best_score_ = -np.inf

    # ---------------------------------------------------------------
    def load_and_split(self, test_size: float = 0.2):
        df = pd.read_csv(self.data_path)
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=RANDOM_STATE, stratify=df['Credit_Score']
        )
        return df_train, df_test

    # ---------------------------------------------------------------
    def run(self, cv_splits: int = 5, tuning_n_iter: int = 10, tuning_cv_splits: int = 3):
        print("=" * 70)
        print("STEP 1: Load data & split train/test")
        print("=" * 70)
        df_train, df_test = self.load_and_split()
        print(f"df_train: {df_train.shape}, df_test: {df_test.shape}")

        print("\n" + "=" * 70)
        print("STEP 2: Preprocessing (fit hanya pada train, transform ke train & test)")
        print("=" * 70)
        X_train, y_train, X_test, y_test = self.preprocessor.fit_transform_split(df_train, df_test)
        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"Target classes (urutan encoding): {list(self.preprocessor.target_classes)}")

        evaluator = Evaluation(self.preprocessor.target_classes)

        print("\n" + "=" * 70)
        print("STEP 3: Eksperimen perbandingan model (5-fold CV, dicatat ke MLflow)")
        print("=" * 70)
        for model_name, estimator in self.MODELS.items():
            with mlflow.start_run(run_name=model_name):
                trainer = Training(model_name, estimator)

                cv_scores = trainer.cross_validate(X_train, y_train, cv_splits=cv_splits)
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))

                # fit pada seluruh X_train untuk evaluasi di test set
                trainer.fit(X_train, y_train)
                y_pred = trainer.predict(X_test)
                test_metrics = evaluator.compute_metrics(y_test, y_pred)

                mlflow.log_param("model_name", model_name)
                mlflow.log_param("cv_splits", cv_splits)
                mlflow.log_metric("cv_f1_macro_mean", cv_mean)
                mlflow.log_metric("cv_f1_macro_std", cv_std)
                for metric_name, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", value)
                mlflow.sklearn.log_model(
                    trainer.pipeline, name="model",
                    serialization_format="pickle"
                )

                self.results_.append({
                    "Model": model_name,
                    "CV_F1_Macro_Mean": cv_mean,
                    "CV_F1_Macro_Std": cv_std,
                    **{f"test_{k}": v for k, v in test_metrics.items()}
                })

                print(f"{model_name:15s} | CV F1-Macro: {cv_mean:.4f} (+/-{cv_std:.4f}) "
                      f"| Test F1-Macro: {test_metrics['f1_macro']:.4f}")

                if cv_mean > self.best_score_:
                    self.best_score_ = cv_mean
                    self.best_model_name_ = model_name

        results_df = pd.DataFrame(self.results_).sort_values("CV_F1_Macro_Mean", ascending=False)
        print("\nRingkasan eksperimen:")
        print(results_df.to_string(index=False))

        print("\n" + "=" * 70)
        print(f"STEP 4: Hyperparameter tuning model terbaik ({self.best_model_name_})")
        print("=" * 70)
        best_estimator = self.MODELS[self.best_model_name_]
        best_trainer = Training(self.best_model_name_, best_estimator)

        with mlflow.start_run(run_name=f"{self.best_model_name_}_tuned"):
            search = best_trainer.tune(
                X_train, y_train,
                param_distributions=self.TUNING_PARAM_DIST,
                n_iter=tuning_n_iter,
                cv_splits=tuning_cv_splits
            )
            y_pred_tuned = best_trainer.predict(X_test)
            tuned_metrics = evaluator.compute_metrics(y_test, y_pred_tuned)
            report_text = evaluator.classification_report_text(y_test, y_pred_tuned)

            mlflow.log_param("model_name", f"{self.best_model_name_}_tuned")
            mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
            for metric_name, value in tuned_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            mlflow.sklearn.log_model(
                best_trainer.pipeline, name="model",
                serialization_format="pickle"
            )

            report_path = os.path.join(self.artifacts_dir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report_text)
            mlflow.log_artifact(report_path)

            print(f"Best params: {search.best_params_}")
            print("\nClassification report (model tuned, test set):")
            print(report_text)

        self.best_trainer_ = best_trainer

        print("\n" + "=" * 70)
        print("STEP 5: Simpan model & preprocessing artifacts")
        print("=" * 70)
        self._save_artifacts(results_df, tuned_metrics, search.best_params_)

        return self.best_trainer_, results_df

    # ---------------------------------------------------------------
    def _save_artifacts(self, results_df, tuned_metrics, best_params):
        model_path = os.path.join(self.artifacts_dir, "model.pkl")
<<<<<<< HEAD
        joblib.dump(self.best_trainer_.pipeline, model_path)
=======
        joblib.dump(self.best_trainer_.pipeline, model_path, compress=9)
>>>>>>> 03bbcfb (First commit dengan Git LFS)

        preprocessing_artifacts = {
            'imputer_values': self.preprocessor.imputer_values_,
            'mix_mapping': self.preprocessor.MIX_MAPPING,
            'kolom_nominal': self.preprocessor.KOLOM_NOMINAL,
            'kolom_final': self.preprocessor.kolom_final_,
            'kolom_hapus': self.preprocessor.KOLOM_HAPUS,
            'kolom_ganti_numerik': self.preprocessor.KOLOM_GANTI_NUMERIK,
            'le_target': self.preprocessor.le_target_,
            'jenis_pinjaman': self.preprocessor.JENIS_PINJAMAN,
        }
        prep_path = os.path.join(self.artifacts_dir, "preprocessing_artifacts.pkl")
        joblib.dump(preprocessing_artifacts, prep_path)

        # Juga simpan satu objek Preprocessing utuh (lebih ringkas dipakai saat inferencing)
        preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")
        joblib.dump(self.preprocessor, preprocessor_path)

        metadata = {
            "best_model_name": self.best_model_name_,
            "best_params": best_params,
            "test_metrics": tuned_metrics,
            "model_comparison": results_df.to_dict(orient="records"),
        }
        metadata_path = os.path.join(self.artifacts_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Model terbaik   : {self.best_model_name_}")
        print(f"Disimpan ke     : {model_path}")
        print(f"Preprocessor    : {preprocessor_path}")
        print(f"Artifacts lain  : {prep_path}")
        print(f"Metadata        : {metadata_path}")


# =============================================================================
# MAIN
# =============================================================================
# CATATAN PENTING:
# File ini SENGAJA tidak menjalankan training saat dieksekusi langsung
# (python pipeline.py). Alasannya: jika class Preprocessing/Training/Evaluation
# di-pickle ketika pipeline.py berjalan sebagai skrip utama (__main__), Python
# mencatat referensi class tersebut sebagai berasal dari module '__main__',
# bukan 'pipeline'. Akibatnya, saat model/preprocessor dimuat ulang dari skrip
# lain (mis. inferencing.py atau app.py Streamlit), proses unpickle akan GAGAL
# dengan error semacam:
#   AttributeError: Can't get attribute 'Preprocessing' on <module '__main__' ...>
#
# Agar referensi module selalu konsisten ('pipeline.Preprocessing', dst),
# jalankan training melalui run_training.py, yang meng-import modul ini
# (bukan menjalankannya langsung):
#
#       python run_training.py
#
if __name__ == "__main__":
    print(
        "File ini tidak dijalankan langsung untuk training.\n"
        "Jalankan training melalui:\n\n"
        "    python run_training.py\n\n"
        "Alasan: lihat komentar pada bagian akhir pipeline.py."
    )

