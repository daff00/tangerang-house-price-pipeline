import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def preprocess_data(df):
    """
    Cleans, filters, and engineers features for the Tangerang house price dataset.
    The function performs the following steps:
    1. Converts data types for price, area, and utility columns.
    2. Extracts the district ('Kecamatan') from the location string.
    3. Filters data to include only relevant properties in Tangerang Regency.
    4. Encodes ordinal and nominal categorical features.
    5. Applies log transformation to skewed numerical features.
    
    Args:
        df (pd.DataFrame): The raw DataFrame scraped from the source.

    Returns:
        pd.DataFrame: The processed DataFrame ready for modeling.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()

    # --- 1. Convert Data Types ---
    
    # Nested helper function for converting price strings (e.g., 'Rp 500 Juta')
    def convert_price(price):
        if not isinstance(price, str):
            return None
        price = price.replace('Rp', '').strip()
        value_str = price.replace(',', '.')
        
        if 'Juta' in value_str:
            value = float(value_str.replace(' Juta', ''))
            return int(value * 1_000_000)
        elif 'Miliar' in value_str:
            value = float(value_str.replace(' Miliar', ''))
            return int(value * 1_000_000_000)
        else:
            return None

    df_processed['Harga'] = df_processed['Harga'].apply(convert_price)

    # Clean and convert other numeric-like columns
    df_processed = df_processed[~df_processed['Daya Listrik'].isin(['Lainnya', 'Lainnya Watt'])]
    df_processed['Daya Listrik'] = df_processed['Daya Listrik'].astype(str).str.replace(' Watt', '', regex=False).astype('Int64')
    df_processed['Luas Tanah'] = df_processed['Luas Tanah'].astype(str).str.replace(' m²', '', regex=False).astype('Int64')
    df_processed['Luas Bangunan'] = df_processed['Luas Bangunan'].astype(str).str.replace(' m²', '', regex=False).astype('Int64')
    
    # Convert remaining numeric columns
    numeric_cols = ['Kamar Tidur', 'Kamar Mandi', 'Jumlah Lantai', 'Carport', 'Kamar Tidur Pembantu', 'Kamar Mandi Pembantu']
    for col in numeric_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').astype('Int64')

    # Standardize 'Kondisi Properti' values before encoding
    df_processed['Kondisi Properti'] = df_processed['Kondisi Properti'].replace({
        'Sudah Renovasi': 'Bagus',
        'Renovasi Total': 'Butuh Renovasi',
        'Renovasi Minimum': 'Butuh Renovasi'
    })

    # --- 2. Feature Extraction ---
    
    # Location mapping dictionary (kept inside the function to make it self-contained)
    desa_kabupaten = {
        "BSD Eminent": "Pagedangan", "BSD Green Wich": "Pagedangan", "Cikupa": "Cikupa", "Cikupa Citra Raya": "Cikupa",
        "Cisauk": "Cisauk", "Suvarna Sutera": "Cikupa", "Panongan": "Panongan", "Curug": "Curug", "Legok": "Legok",
        "Pasar Kemis": "Pasar Kemis", "Sepatan": "Sepatan", "Bitung": "Curug", "Tigaraksa": "Tigaraksa", "Balaraja": "Balaraja",
        "Jayanti": "Jayanti", "Kosambi": "Kosambi", "Teluk Naga": "Teluk Naga", "Mauk": "Mauk", "Kresek": "Kresek",
        "Solear": "Solear", "Sindang Jaya": "Sindang Jaya", "Rajeg": "Rajeg", "Gading Serpong": "Kelapa Dua",
        # Add any other mappings here
    }

    def extract_kecamatan(lokasi):
        if not isinstance(lokasi, str):
            return None
        for key, value in desa_kabupaten.items():
            if key in lokasi:
                return value
        return None

    df_processed['Kecamatan'] = df_processed['Lokasi'].apply(extract_kecamatan)
    df_processed = df_processed.drop(columns=['Lokasi']) # Drop original location column

    # --- 3. Filtering Data ---
    
    # Drop rows where Kecamatan could not be identified or that are outside the scope
    df_processed.dropna(subset=['Kecamatan', 'Harga'], inplace=True)
    
    # Apply domain-specific filters
    df_processed = df_processed[df_processed['Carport'].between(1, 4)]
    df_processed = df_processed[df_processed['Jumlah Lantai'].between(1, 3)]
    df_processed = df_processed[~df_processed['Sertifikat'].isin(["Strata", "Hak Sewa"])]
    df_processed = df_processed[df_processed['Luas Bangunan'] <= 1000]
    df_processed = df_processed[df_processed['Luas Tanah'] <= 1000]

    # --- 5. Encoding --- (Note: Swapped order with Outliers as encoding is on original values)

    # Ordinal Encoding
    ordinal_features = {
        'Sertifikat': ['PPJB', 'HGB', 'Lainnya', 'SHM'],
        'Kondisi Properti': ['Butuh Renovasi', 'Bagus', 'Baru'],
        'Daya Listrik': sorted(df_processed['Daya Listrik'].dropna().unique().tolist())
    }
    for feature, categories in ordinal_features.items():
        encoder = OrdinalEncoder(categories=[categories])
        df_processed[[feature]] = encoder.fit_transform(df_processed[[feature]])

    # One-Hot Encoding for 'Kecamatan'
    df_processed = pd.get_dummies(df_processed, columns=['Kecamatan'], prefix='kec')

    # --- 4. Outliers Handling --- (Log transform is often done after filtering/before scaling)

    # Log transform skewed features to handle outliers and normalize distribution
    for col in ['Harga', 'Luas Tanah', 'Luas Bangunan']:
        df_processed[col] = np.log1p(df_processed[col])

    return df_processed