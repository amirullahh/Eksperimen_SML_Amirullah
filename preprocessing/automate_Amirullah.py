"""
automate_Amirullah.py
Automated preprocessing pipeline for House Prices dataset
Author: Amirullah
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

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_data(filepath):
    """Load raw dataset"""
    print(f"\n{'='*60}")
    print("LOADING DATASET")
    print(f"{'='*60}")
    
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape}")
    print(f"   - Rows: {df.shape[0]}")
    print(f"   - Columns: {df.shape[1]}")
    return df


def handle_missing_values(df):
    """Handle missing values based on data context"""
    print(f"\n{'='*60}")
    print("HANDLING MISSING VALUES")
    print(f"{'='*60}")
    
    # Categorical features: NA means "None"
    categorical_na_none = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    
    for col in categorical_na_none:
        if col in df.columns:
            df[col].fillna('None', inplace=True)
    
    print(f"✅ Categorical features filled with 'None'")
    
    # Numerical features: NA means 0
    numerical_na_zero = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    
    for col in numerical_na_zero:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    print(f"✅ Numerical features filled with 0")
    
    # LotFrontage: Fill with median per Neighborhood
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        print(f"✅ LotFrontage filled with neighborhood median")
    
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
    
    print(f"✅ Total missing values: {df.isnull().sum().sum()}")
    
    return df


def feature_engineering(df):
    """Create new features"""
    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING")
    print(f"{'='*60}")
    
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
    
    print(f"✅ Created new features")
    print(f"   - Total features: {df.shape[1]}")
    
    return df


def remove_outliers(df):
    """Remove extreme outliers"""
    print(f"\n{'='*60}")
    print("REMOVING OUTLIERS")
    print(f"{'='*60}")
    
    original_shape = df.shape[0]
    
    # Remove outliers
    df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))]
    df = df[df['LotArea'] <= 100000]
    df = df[df['TotalBsmtSF'] <= 6000]
    
    removed = original_shape - df.shape[0]
    print(f"✅ Removed {removed} outliers ({removed/original_shape*100:.2f}%)")
    
    return df


def log_transform(df):
    """Apply log transformation"""
    print(f"\n{'='*60}")
    print("LOG TRANSFORMATION")
    print(f"{'='*60}")
    
    # Log transform target
    df['SalePrice'] = np.log1p(df['SalePrice'])
    print(f"✅ SalePrice log-transformed")
    
    # Find skewed features
    numerical_features = df.select_dtypes(include=[np.number]).columns
    numerical_features = numerical_features.drop(['Id', 'SalePrice'])
    
    skewed_features = df[numerical_features].apply(lambda x: skew(x))
    high_skew = skewed_features[abs(skewed_features) > 0.5]
    
    # Log transform skewed features
    for feature in high_skew.index:
        df[feature] = np.log1p(df[feature])
    
    print(f"✅ Log-transformed {len(high_skew)} skewed features")
    
    return df


def encode_categorical(df):
    """Encode categorical features"""
    print(f"\n{'='*60}")
    print("ENCODING CATEGORICAL FEATURES")
    print(f"{'='*60}")
    
    # Ordinal encoding for quality features
    ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                        'HeatingQC', 'KitchenQual', 'FireplaceQu', 
                        'GarageQual', 'GarageCond', 'PoolQC']
    
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    
    for feature in ordinal_features:
        if feature in df.columns:
            df[feature] = df[feature].map(quality_map)
    
    print(f"✅ Ordinal features encoded")
    
    # One-hot encoding for nominal features
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    nominal_features = [f for f in categorical_features if f not in ordinal_features]
    
    if nominal_features:
        df = pd.get_dummies(df, columns=nominal_features, drop_first=True)
        print(f"✅ One-hot encoded {len(nominal_features)} nominal features")
    
    print(f"✅ Total features after encoding: {df.shape[1]}")
    
    return df


def split_and_scale(df):
    """Split data and apply scaling"""
    print(f"\n{'='*60}")
    print("TRAIN-TEST SPLIT & SCALING")
    print(f"{'='*60}")
    
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
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"✅ Features scaled using RobustScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_data(X_train, X_test, y_train, y_test, scaler):
    """Save preprocessed data and scaler"""
    print(f"\n{'='*60}")
    print("SAVING PREPROCESSED DATA")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Combine features and target
    train_processed = X_train.copy()
    train_processed['SalePrice'] = y_train
    
    test_processed = X_test.copy()
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
# MAIN PIPELINE
# ============================================================

def main():
    """Main preprocessing pipeline"""
    print("\n" + "="*60)
    print("AUTOMATED PREPROCESSING PIPELINE")
    print("House Prices Dataset - Amirullah")
    print("="*60)
    
    try:
        # Load data
        df = load_data(RAW_DATA_PATH)
        
        # Preprocessing steps
        df = handle_missing_values(df)
        df = feature_engineering(df)
        df = remove_outliers(df)
        df = log_transform(df)
        df = encode_categorical(df)
        
        # Split and scale
        X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
        
        # Save
        save_data(X_train, X_test, y_train, y_test, scaler)
        
        # Summary
        print(f"\n{'='*60}")
        print("🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"\n📊 Final Dataset Shape:")
        print(f"   - Training: {X_train.shape}")
        print(f"   - Test: {X_test.shape}")
        print(f"   - Total Features: {len(X_train.columns)}")
        print(f"\n✅ Data ready for modeling!\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()
