"""
run_training.py
=================
Entry point resmi untuk menjalankan training pipeline (pipeline.py).

PENTING - kenapa file ini ada (bukan langsung `python pipeline.py`):
----------------------------------------------------------------------
pipeline.py berisi definisi class custom (Preprocessing, Training, Evaluation,
ExperimentRunner). Jika pipeline.py dijalankan langsung sebagai skrip utama
(python pipeline.py), Python akan mencatat referensi class tersebut sebagai
berasal dari module '__main__'. Akibatnya, saat model/preprocessor (.pkl)
dimuat ulang dari skrip LAIN (inferencing.py, app.py Streamlit), proses
unpickle akan GAGAL dengan error:

    AttributeError: Can't get attribute 'Preprocessing' on <module '__main__' ...>

Dengan menjalankan training melalui run_training.py (yang meng-import
pipeline.py sebagai modul, bukan menjalankannya langsung), referensi class
tercatat secara konsisten sebagai 'pipeline.Preprocessing', dst — sehingga
bisa dimuat ulang dari script manapun tanpa error.

Cara menjalankan:
------------------
    python run_training.py

Hasil:
------
    - model.pkl                     -> model terbaik (Random Forest, tuned)
    - preprocessor.pkl              -> objek Preprocessing yang sudah di-fit
    - preprocessing_artifacts.pkl   -> median/modus/mapping/kolom (alternatif granular)
    - metadata.json                 -> ringkasan model terbaik & parameter
    - classification_report.txt     -> laporan evaluasi model tuned
    - mlruns/                       -> tracking eksperimen MLflow (5 model + tuning)

Semua file di atas disimpan LANGSUNG di folder ini (bukan di subfolder
artifacts/), supaya cocok dengan path yang dipakai inferencing.py dan app.py.

Catatan: data_D.csv harus berada di folder yang sama dengan file ini.
Training penuh (5 model x 5-fold CV + tuning 10 iterasi x 3-fold pada
~20.000 baris) memakan waktu sekitar 7-10 menit tergantung spesifikasi komputer.
"""

from pipeline import ExperimentRunner

if __name__ == "__main__":
    runner = ExperimentRunner(
        data_path="data_D.csv",
        experiment_name="Credit_Score_Classification",
        artifacts_dir=".",          # simpan langsung di folder ini, bukan di artifacts/
        tracking_uri="./mlruns"
    )
    best_trainer, results_df = runner.run(
        cv_splits=5,
        tuning_n_iter=10,
        tuning_cv_splits=3
    )
    print(
        "\nTraining selesai. File model.pkl dan preprocessor.pkl sudah siap dipakai "
        "oleh inferencing.py / app.py.\n"
        "Untuk melihat hasil eksperimen di MLflow UI, jalankan:\n\n"
        "    (Windows PowerShell)\n"
        "    $env:MLFLOW_ALLOW_FILE_STORE = \"true\"\n"
        "    mlflow ui --backend-store-uri ./mlruns\n"
    )
