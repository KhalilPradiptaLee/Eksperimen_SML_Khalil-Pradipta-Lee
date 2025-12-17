import pandas as pd
import numpy as np
import os

# =========================
# Konfigurasi
# =========================
url = "https://drive.google.com/uc?id=1mEOFj5EuMjcWTmieXEooFSUIe2mGyEta"
output_name = "dataset_mesin_membangun_sistem_machine_learning_preprocessing.csv"
output_dir = "preprocessing"


def preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Melakukan preprocessing data Student Performance secara otomatis
    dan mengembalikan data yang siap dilatih.
    """

    # =========================
    # 1. Load dataset
    # =========================
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error saat memuat dataset: {e}")
        return pd.DataFrame()

    print("1. Dataset berhasil dimuat.")

    np.random.seed(42)

    # =========================
    # 2. Drop kolom tidak penting
    # =========================
    df = df.drop(columns=['StudentID', 'Name', 'Gender', 'Study Hours', 'Attendance (%)'], errors='ignore')
    print("2. Kolom identifier berhasil dihapus.")

    # =========================
    # 3. Imputasi kategorikal (random)
    # =========================

    if 'ParentalSupport' in df.columns:
        df.loc[df['ParentalSupport'].isna(), 'ParentalSupport'] = np.random.choice(
            ['Low', 'Medium', 'High'],
            size=df['ParentalSupport'].isna().sum()
        )

    if 'Online Classes Taken' in df.columns:
        df.loc[df['Online Classes Taken'].isna(), 'Online Classes Taken'] = np.random.choice(
            [True, False],
            size=df['Online Classes Taken'].isna().sum()
        )

    print("3. Imputasi data kategorikal berhasil.")

    # =========================
    # 4. Imputasi numerik (mean + ceil)
    # =========================
    kolom_numerik = [
        'AttendanceRate',
        'StudyHoursPerWeek',
        'PreviousGrade',
        'FinalGrade',
        'Study Hours',
        'Attendance (%)'
    ]

    for col in kolom_numerik:
        if col in df.columns:
            df[col] = df[col].fillna(np.ceil(df[col].mean()))

    print("4. Imputasi data numerik berhasil.")

    # =========================
    # 5. Encoding kategorikal
    # =========================
    if 'ParentalSupport' in df.columns:
        df['ParentalSupport'] = df['ParentalSupport'].map({
            'Low': 0,
            'Medium': 1,
            'High': 2
        })

    if 'Online Classes Taken' in df.columns:
        df['Online Classes Taken'] = df['Online Classes Taken'].map({
            False: 0,
            True: 1
        })

    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({
            'Male': 0,
            'Female': 1
        })

    print("5. Encoding data kategorikal berhasil.")

    print(f"6. Preprocessing selesai. Total data: {len(df)}")

    return df


# =========================
# Eksekusi langsung
# =========================
if __name__ == "__main__":

    os.makedirs(output_dir, exist_ok=True)

    print("Memulai proses preprocessing Student Performance...")

    df_clean = preprocess_data(url)

    if not df_clean.empty:
        output_path = os.path.join(output_dir, output_name)
        df_clean.to_csv(output_path, index=False)
        print(f"Data siap dilatih disimpan di: {output_path}") 

