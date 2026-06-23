"""
app.py
=======
Aplikasi web Streamlit untuk deployment model Credit Score Classification.

Pengguna mengisi form berisi data nasabah (sesuai fitur mentah pada dataset
asli), lalu aplikasi memanggil inferencing.py untuk menjalankan preprocessing
+ prediksi, dan menampilkan hasilnya (kelas prediksi + probabilitas tiap kelas).

Cara menjalankan secara lokal:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from inferencing import load_artifacts, predict_single


# -----------------------------------------------------------------------
# Konfigurasi halaman
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Score Prediction")
st.markdown(
    "Aplikasi ini memprediksi **Credit Score** nasabah (Good / Standard / Poor) "
    "berdasarkan data finansial, menggunakan model Random Forest yang telah dilatih "
    "dan dicatat eksperimennya dengan MLflow."
)


# -----------------------------------------------------------------------
# Load model & preprocessor (di-cache supaya tidak load ulang setiap interaksi)
# -----------------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    model, preprocessor = get_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(
        "Gagal memuat model/preprocessor. Pastikan file `model.pkl` dan "
        f"`preprocessor.pkl` ada di folder yang sama dengan app.py.\n\nDetail error: {e}"
    )


# -----------------------------------------------------------------------
# Daftar pilihan kategori (diambil dari kategori valid pada data training)
# -----------------------------------------------------------------------
OCCUPATION_OPTIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
    "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician",
    "Scientist", "Teacher", "Writer"
]

CREDIT_MIX_OPTIONS = ["Bad", "Standard", "Good"]
PAYMENT_MIN_AMOUNT_OPTIONS = ["Yes", "No"]
SPENDING_OPTIONS = ["High", "Low"]
PAYMENT_SIZE_OPTIONS = ["Small", "Medium", "Large"]

LOAN_TYPE_OPTIONS = [
    "Auto Loan", "Student Loan", "Credit-Builder Loan", "Personal Loan",
    "Home Equity Loan", "Mortgage Loan", "Payday Loan", "Debt Consolidation Loan"
]


# -----------------------------------------------------------------------
# Form input
# -----------------------------------------------------------------------
st.header("📋 Data Nasabah")

with st.form("credit_score_form"):

    st.subheader("Data Pribadi & Pekerjaan")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (umur)", min_value=18, max_value=100, value=32)
    with col2:
        occupation = st.selectbox("Occupation (pekerjaan)", OCCUPATION_OPTIONS, index=3)

    st.subheader("Pendapatan")
    col1, col2 = st.columns(2)
    with col1:
        annual_income = st.number_input(
            "Annual Income (pendapatan tahunan)", min_value=0.0, value=56125.5, step=1000.0
        )
    with col2:
        monthly_inhand_salary = st.number_input(
            "Monthly Inhand Salary (gaji bulanan diterima)", min_value=0.0, value=4875.13, step=100.0
        )

    st.subheader("Rekening & Kartu Kredit")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_bank_accounts = st.number_input("Jumlah Rekening Bank", min_value=0, max_value=50, value=8)
    with col2:
        num_credit_card = st.number_input("Jumlah Kartu Kredit", min_value=0, max_value=50, value=3)
    with col3:
        interest_rate = st.number_input("Interest Rate (%)", min_value=1, max_value=100, value=18)

    st.subheader("Pinjaman")
    col1, col2 = st.columns(2)
    with col1:
        num_of_loan = st.number_input("Jumlah Pinjaman", min_value=0, max_value=100, value=2)
    with col2:
        type_of_loan = st.multiselect(
            "Jenis Pinjaman (boleh pilih lebih dari satu)",
            LOAN_TYPE_OPTIONS,
            default=["Credit-Builder Loan", "Mortgage Loan"]
        )

    st.subheader("Riwayat Pembayaran")
    col1, col2 = st.columns(2)
    with col1:
        delay_from_due_date = st.number_input(
            "Delay from Due Date (hari rata-rata telat bayar)", min_value=0, max_value=100, value=30
        )
    with col2:
        num_of_delayed_payment = st.number_input(
            "Jumlah Pembayaran Terlambat", min_value=0, max_value=100, value=14
        )

    col1, col2 = st.columns(2)
    with col1:
        changed_credit_limit = st.number_input(
            "Changed Credit Limit (% perubahan limit kredit)", value=17.89, step=0.1
        )
    with col2:
        num_credit_inquiries = st.number_input(
            "Jumlah Credit Inquiries", min_value=0.0, max_value=100.0, value=4.0, step=1.0
        )

    st.subheader("Kondisi Kredit")
    col1, col2 = st.columns(2)
    with col1:
        credit_mix = st.selectbox("Credit Mix", CREDIT_MIX_OPTIONS, index=1)
    with col2:
        outstanding_debt = st.number_input(
            "Outstanding Debt (sisa hutang)", min_value=0.0, value=370.22, step=10.0
        )

    col1, col2 = st.columns(2)
    with col1:
        credit_utilization_ratio = st.number_input(
            "Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=32.01, step=0.1
        )
    with col2:
        credit_history_years = st.number_input(
            "Credit History - Tahun", min_value=0, max_value=50, value=28
        )
    credit_history_months = st.number_input(
        "Credit History - Bulan tambahan", min_value=0, max_value=11, value=10
    )

    payment_of_min_amount = st.selectbox(
        "Payment of Min Amount (apakah hanya bayar minimum?)", PAYMENT_MIN_AMOUNT_OPTIONS, index=0
    )

    st.subheader("Pengeluaran & Investasi")
    col1, col2 = st.columns(2)
    with col1:
        total_emi_per_month = st.number_input(
            "Total EMI per Month (cicilan bulanan)", min_value=0.0, value=81.82, step=10.0
        )
    with col2:
        amount_invested_monthly = st.number_input(
            "Amount Invested Monthly (investasi bulanan)", min_value=0.0, value=182.07, step=10.0
        )

    col1, col2 = st.columns(2)
    with col1:
        spending_behavior = st.selectbox("Spending Behavior", SPENDING_OPTIONS, index=0)
    with col2:
        payment_size = st.selectbox("Payment Size", PAYMENT_SIZE_OPTIONS, index=1)

    monthly_balance = st.number_input(
        "Monthly Balance (saldo akhir bulan)", value=473.62, step=10.0
    )

    submitted = st.form_submit_button("🔍 Prediksi Credit Score", use_container_width=True)


# -----------------------------------------------------------------------
# Proses prediksi setelah form di-submit
# -----------------------------------------------------------------------
if submitted:
    if not artifacts_loaded:
        st.error("Model belum berhasil dimuat, prediksi tidak bisa dilakukan.")
    else:
        # Susun ulang Payment_Behaviour dari Spending_Behavior + Payment_Size,
        # mengikuti format kategori asli pada data training
        payment_behaviour = f"{spending_behavior}_spent_{payment_size}_value_payments"

        # Susun Type_of_Loan jadi satu string, mengikuti format asli
        # (contoh asli: "Credit-Builder Loan, and Mortgage Loan")
        if len(type_of_loan) == 0:
            type_of_loan_str = "Not Specified"
        elif len(type_of_loan) == 1:
            type_of_loan_str = type_of_loan[0]
        else:
            type_of_loan_str = ", ".join(type_of_loan[:-1]) + ", and " + type_of_loan[-1]

        credit_history_age_str = f"{int(credit_history_years)} Years and {int(credit_history_months)} Months"

        input_data = {
            "Age": age,
            "Occupation": occupation,
            "Annual_Income": annual_income,
            "Monthly_Inhand_Salary": monthly_inhand_salary,
            "Num_Bank_Accounts": num_bank_accounts,
            "Num_Credit_Card": num_credit_card,
            "Interest_Rate": interest_rate,
            "Num_of_Loan": num_of_loan,
            "Type_of_Loan": type_of_loan_str,
            "Delay_from_due_date": delay_from_due_date,
            "Num_of_Delayed_Payment": num_of_delayed_payment,
            "Changed_Credit_Limit": changed_credit_limit,
            "Num_Credit_Inquiries": num_credit_inquiries,
            "Credit_Mix": credit_mix,
            "Outstanding_Debt": outstanding_debt,
            "Credit_Utilization_Ratio": credit_utilization_ratio,
            "Credit_History_Age": credit_history_age_str,
            "Payment_of_Min_Amount": payment_of_min_amount,
            "Total_EMI_per_month": total_emi_per_month,
            "Amount_invested_monthly": amount_invested_monthly,
            "Payment_Behaviour": payment_behaviour,
            "Monthly_Balance": monthly_balance,
        }

        with st.spinner("Menjalankan prediksi..."):
            try:
                hasil = predict_single(input_data, model, preprocessor)

                st.header("📊 Hasil Prediksi")

                label = hasil["predicted_class"]
                if label == "Good":
                    st.success(f"### Credit Score: **{label}** ✅")
                elif label == "Standard":
                    st.warning(f"### Credit Score: **{label}** ⚠️")
                else:
                    st.error(f"### Credit Score: **{label}** ❌")

                if hasil["probabilities"]:
                    st.subheader("Probabilitas tiap kelas")
                    proba_df = pd.DataFrame(
                        list(hasil["probabilities"].items()),
                        columns=["Kelas", "Probabilitas"]
                    ).sort_values("Probabilitas", ascending=False)
                    proba_df["Probabilitas"] = proba_df["Probabilitas"].apply(lambda x: f"{x:.2%}")
                    st.dataframe(proba_df, hide_index=True, use_container_width=True)

                with st.expander("Lihat data input yang dikirim ke model"):
                    st.json(input_data)

            except Exception as e:
                st.error(f"Terjadi error saat melakukan prediksi: {e}")


st.markdown("---")
st.caption("Final Project DTSC6012001 - Model Deployment | Credit Score Classification")
