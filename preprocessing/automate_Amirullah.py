"""
automate_Amirullah.py
Automated preprocessing pipeline for House Prices dataset
Author: Amirullah Hidayat
Date: February 2026
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew
import os
import pickle

# ============================================================
# CONFIGURATION
# ============================================================
RAW_DATA_PATH = '../dataset_raw/train.csv'
OUTPUT_DIR = 'dataset_preprocessing'
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("=" * 60)
print("AUTOMATED PREPROCESSING PIPELINE")
print("House Prices Dataset - Amirullah")
print("=" * 60)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/8] Loading data...")
try:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"✅ Data loaded: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: File not found at {RAW_DATA_PATH}")
    print(f"   Please check if train.csv exists in dataset_raw folder")
    exit(1)

# ============================================================
# STEP 2: HANDLE MISSING VALUES
# ============================================================
print("\n[2/8] Handling missing values...")

# Categorical: NA means "None"
categorical_na_none = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]
for col in categorical_na_none:
    if col in df.columns:
        df[col].fillna('None', inplace=True)

# Numerical: NA means 0
numerical_na_zero = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]
for col in numerical_na_zero:
    if col in df.columns:
        df[col].fillna(0, inplace=True)

# LotFrontage: Fill with neighborhood median
if 'LotFrontage' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )

# Remaining categorical: Fill with mode
remaining_categorical = df.select_dtypes(include=['object']).columns
for col in remaining_categorical:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Remaining numerical: Fill with median
remaining_numerical = df.select_dtypes(include=[np.number]).columns
for col in remaining_numerical:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

print(f"✅ Missing values handled: {df.isnull().sum().sum()} remaining")

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
print("\n[3/8] Feature engineering...")

# Total Square Footage
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# Total Bathrooms
df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                   df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])

# Age features
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# Binary features
df['HasPool'] = (df['PoolArea'] > 0).astype(int)
df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

# Interaction features
df['OverallQual_x_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
df['OverallQual_x_TotalSF'] = df['OverallQual'] * df['TotalSF']

# Total Porch
df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] + 
                      df['3SsnPorch'] + df['ScreenPorch'])

print(f"✅ Feature engineering completed: {df.shape[1]} features")

# ============================================================
# STEP 4: REMOVE OUTLIERS
# ============================================================
print("\n[4/8] Removing outliers...")

original_shape = df.shape[0]

# Remove outliers
df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))]
df = df[df['LotArea'] <= 100000]
df = df[df['TotalBsmtSF'] <= 6000]

removed = original_shape - df.shape[0]
print(f"✅ Removed {removed} outliers ({removed/original_shape*100:.2f}%)")

# ============================================================
# STEP 5: LOG TRANSFORMATION
# ============================================================
print("\n[5/8] Log transformation...")

# Log transform target
df['SalePrice'] = np.log1p(df['SalePrice'])

# Find skewed features
numerical_features = df.select_dtypes(include=[np.number]).columns
numerical_features = numerical_features.drop(['Id', 'SalePrice'])

skewed_features = df[numerical_features].apply(lambda x: skew(x))
high_skew = skewed_features[abs(skewed_features) > 0.5]

# Log transform skewed features
for feature in high_skew.index:
    df[feature] = np.log1p(df[feature])

print(f"✅ Log transformed {len(high_skew)} skewed features")

# ============================================================
# STEP 6: CATEGORICAL ENCODING
# ============================================================
print("\n[6/8] Encoding categorical features...")

# Ordinal encoding
ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                    'HeatingQC', 'KitchenQual', 'FireplaceQu', 
                    'GarageQual', 'GarageCond', 'PoolQC']

quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

for feature in ordinal_features:
    if feature in df.columns:
        df[feature] = df[feature].map(quality_map)

# One-hot encoding
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
nominal_features = [f for f in categorical_features if f not in ordinal_features]

if nominal_features:
    df = pd.get_dummies(df, columns=nominal_features, drop_first=True)

print(f"✅ Encoding completed: {df.shape[1]} features")

# ============================================================
# STEP 7: TRAIN-TEST SPLIT & SCALING
# ============================================================
print("\n[7/8] Train-test split and scaling...")

# Separate features and target
X = df.drop(['SalePrice', 'Id'], axis=1)
y = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f"✅ Features scaled using RobustScaler")

# ============================================================
# STEP 8: SAVE PROCESSED DATA
# ============================================================
print("\n[8/8] Saving processed data...")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Combine features and target
train_processed = X_train_scaled.copy()
train_processed['SalePrice'] = y_train

test_processed = X_test_scaled.copy()
test_processed['SalePrice'] = y_test

# Save to CSV
train_processed.to_csv(f'{OUTPUT_DIR}/train_processed.csv', index=False)
test_processed.to_csv(f'{OUTPUT_DIR}/test_processed.csv', index=False)

# Save scaler
with open(f'{OUTPUT_DIR}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Train data saved: {OUTPUT_DIR}/train_processed.csv")
print(f"✅ Test data saved: {OUTPUT_DIR}/test_processed.csv")
print(f"✅ Scaler saved: {OUTPUT_DIR}/scaler.pkl")

# ============================================================
# COMPLETION
# ============================================================
print("\n" + "=" * 60)
print("🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📊 Final Dataset Shape:")
print(f"   - Training: {train_processed.shape}")
print(f"   - Test: {test_processed.shape}")
print(f"   - Total Features: {len(X_train.columns)}")
print(f"\n✅ Data ready for modeling!\n")