\# Eksperimen SML - Amirullah



\## 📊 Project Overview

Proyek \*\*Machine Learning System\*\* untuk prediksi harga rumah (House Prices) menggunakan dataset dari Kaggle Competition: \[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)



\## 🎯 Tujuan

Membangun \*\*automated preprocessing pipeline\*\* yang dapat:

\- ✅ Melakukan EDA (Exploratory Data Analysis)

\- ✅ Handling missing values secara otomatis

\- ✅ Feature engineering

\- ✅ Outliers removal

\- ✅ Data transformation (log, scaling)

\- ✅ Categorical encoding

\- ✅ Train-test split



\## 📁 Struktur Folder



```bash

Eksperimen\_SML\_Amirullah/

├── .github/

│ └── workflows/

│ └── preprocessing.yml # GitHub Actions workflow

├── dataset\_raw/

│ └── train.csv # Raw dataset dari Kaggle

├── preprocessing/

│ ├── Eksperimen\_Amirullah.ipynb # Notebook eksperimen lengkap

│ ├── automate\_Amirullah.py # Automation script

│ └── dataset\_preprocessing/ # Output folder

│ ├── train\_processed.csv

│ ├── test\_processed.csv

│ └── scaler.pkl

├── .gitignore

└── README.md

```bash





\## 🚀 Quick Start



\### Prerequisites

\- Python 3.12.7

\- pandas, numpy, scikit-learn, scipy



\### Installation

```bash

pip install pandas numpy scikit-learn scipy

```bash



```bash

cd preprocessing

python automate\_Amirullah.py

```bash



📊 Dataset Information

Source: Kaggle - House Prices Advanced Regression Techniques



Original Dataset:



Training samples: 1,460

Features: 79 (numerical + categorical)

Target: SalePrice



After Preprocessing:



Training samples: 1,163 (~20% outliers removed)

Test samples: 291 (20% split)

Features: 240 (after one-hot encoding)

Target: log(SalePrice)



🔧 Preprocessing Steps



1. Missing Values Handling



Categorical: Fill with 'None'

Numerical: Fill with 0 or median



2\. Feature Engineering



TotalSF, TotalBath, HouseAge, etc.

Binary features (HasPool, HasGarage, etc.)



3\. Interaction features



Outliers Removal

Remove extreme values in GrLivArea, LotArea, TotalBsmtSF



4\. Log Transformation



Target variable (SalePrice)

Skewed numerical features



5\. Categorical Encoding



Ordinal: Label encoding for quality features

Nominal: One-hot encoding



6\. Scaling



RobustScaler (robust to outliers)



🤖 GitHub Actions

Repository ini menggunakan GitHub Actions untuk automated preprocessing setiap kali ada push/pull request.



Workflow akan:



✅ Setup Python environment

✅ Install dependencies

✅ Run preprocessing script

✅ Upload preprocessed data sebagai artifacts



👤 Author

Amirullah Hidayat



Dicoding Submission: Membangun Sistem Machine Learning



Date: February 2026



📝 License

This project is for educational purposes (Dicoding submission).





