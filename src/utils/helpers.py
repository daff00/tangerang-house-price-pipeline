import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_validate, cross_val_predict, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import optuna
import warnings
warnings.filterwarnings('ignore')

# Constants
DESA_KABUPATEN_MAPPING = {
    "BSD Eminent": "Pagedangan", "BSD Green Wich": "Pagedangan", "BSD Avani": "Pagedangan",
    "BSD Vanya Park": "Pagedangan", "BSD Foresta": "Pagedangan", "BSD The Green": "Pagedangan",
    "BSD Kencana Loka": "Pagedangan", "BSD Taman Giri Loka": "Pagedangan", "BSD Telaga Golf": "Pagedangan", "BSD Neo Catalonia": "Pagedangan", "BSD Provance Parkland": "Pagedangan", "BSD Alegria": "Pagedangan", "Cikupa": "Cikupa", "Cikupa Citra Raya": "Cikupa", "Cisauk": "Cisauk", "Suvarna Sutera": "Cikupa", "Panongan": "Panongan", "Curug": "Curug", "Legok": "Legok", "Pasar Kemis": "Pasar Kemis", "Sepatan": "Sepatan", "Bitung": "Curug", "Tigaraksa": "Tigaraksa", "Balaraja": "Balaraja", "Jayanti": "Jayanti", "Kosambi": "Kosambi", "Teluk Naga": "Teluk Naga", "Mauk": "Mauk", "Kresek": "Kresek", "Solear": "Solear", "Sindang Jaya": "Sindang Jaya", "Rajeg": "Rajeg", "Kadu": "Kadu", "Jatake": "Jatiuwung", "Bojong Nangka": "Kelapa Dua", "Suradita": "Cisauk", "Cukang Galih": "Curug", "Gading Serpong": "Kelapa Dua", "Gading Serpong Pondok Hijau Golf": "Kelapa Dua", "Gading Serpong The Spring": "Kelapa Dua", "Gading Serpong Scientia Garden": "Kelapa Dua", "Gading Serpong Samara Village": "Kelapa Dua", "Gading Serpong IL Lago": "Kelapa Dua", "Gading Serpong Elista Village": "Kelapa Dua", "Gading Serpong Serenade Lake": "Kelapa Dua", "Gading Serpong Cluster Bohemia": "Kelapa Dua", "Gading Serpong Omaha Village": "Kelapa Dua", "Gading Serpong La Bella Village": "Kelapa Dua", "Gading Serpong Virginia Village": "Kelapa Dua", "Gading Serpong Cluster Oleaster": "Kelapa Dua", "Gading Serpong Cluster Michelia": "Kelapa Dua", "Gading Serpong Karelia Village": "Kelapa Dua", "Gading Serpong Andalucia": "Kelapa Dua", "Gading Serpong Cluster IL Rosa": "Kelapa Dua", "Pasar Kemis": "Pasar Kemis" # and more
}

ENCODING_CATEGORIES = {
    'sertifikat': ['PPJB', 'HGB', 'Lainnya', 'SHM'],
    'kondisi': ['Butuh Renovasi', 'Bagus', 'Baru'],
    'daya_listrik': [450, 900, 1300, 2200, 3300, 3500, 4400, 5500, 6600, 7600, 7700, 
                    8000, 10000, 10600, 11000, 13200, 16500, 22000, 23000, 30500, 33000]
}

class DataPreprocessor:
    """Class for preprocessing house price data"""
    
    @staticmethod
    def convert_price(price):
        """Convert price string to numeric value"""
        if pd.isna(price) or price is None:
            return None
            
        price_str = str(price)
        price_str = price_str.replace('Rp', '').strip()
        
        if 'Juta' in price_str:
            value = price_str.replace(' Juta', '').replace(',', '.')
            multiplier = 1e6
        elif 'Miliar' in price_str:
            value = price_str.replace(' Miliar', '').replace(',', '.')
            multiplier = 1e9
        else:
            return None
        
        try:
            value = float(value)
            return int(value * multiplier)
        except ValueError:
            return None
    
    @staticmethod
    def clean_numeric_columns(df):
        """Clean and convert numeric columns"""
        df_clean = df.copy()
        
        # Remove rows with invalid 'Daya Listrik' values
        df_clean = df_clean[~df_clean['Daya Listrik'].isin(['Lainnya', 'Lainnya Watt'])]
        
        # Clean and convert columns
        numeric_columns = {
            'Daya Listrik': ' Watt',
            'Luas Tanah': ' m²',
            'Luas Bangunan': ' m²'
        }
        
        for col, suffix in numeric_columns.items():
            df_clean[col] = (df_clean[col].astype(str)
                            .str.replace(suffix, '', regex=False)
                            .astype('Int64'))
        
        # Convert other integer columns
        int_columns = ['Kamar Tidur', 'Kamar Mandi', 'Jumlah Lantai', 'Carport', 
                      'Kamar Tidur Pembantu', 'Kamar Mandi Pembantu']
        
        for col in int_columns:
            df_clean[col] = df_clean[col].astype('Int64')
        
        return df_clean
    
    @staticmethod
    def clean_condition_column(df):
        """Clean and standardize condition column"""
        df_clean = df.copy()
        
        condition_mapping = {
            'Sudah Renovasi': 'Bagus',
            'Renovasi Total': 'Butuh Renovasi', 
            'Renovasi Minimum': 'Butuh Renovasi'
        }
        
        df_clean['Kondisi Properti'] = (df_clean['Kondisi Properti']
                                       .replace(condition_mapping))
        
        return df_clean
    
    @staticmethod
    def extract_kecamatan(lokasi):
        """Extract subdistrict from location string"""
        if pd.isna(lokasi):
            return None
            
        for key in DESA_KABUPATEN_MAPPING.keys():
            if key in str(lokasi):
                return DESA_KABUPATEN_MAPPING[key]
        return None
    
    @staticmethod
    def filter_tangerang_regency(df):
        """Filter data for Tangerang Regency only"""
        df_filtered = df.copy()
        df_filtered["Kecamatan"] = df_filtered["Lokasi"].apply(DataPreprocessor.extract_kecamatan)
        df_filtered = df_filtered.dropna(subset=["Kecamatan"])
        df_filtered = df_filtered.drop(columns=['Lokasi'])
        return df_filtered
    
    @staticmethod
    def apply_filters(df):
        """Apply business logic filters to dataset"""
        df_filtered = df.copy()
        
        # Apply filters
        df_filtered = df_filtered[(df_filtered['Carport'] != 0) & (df_filtered['Carport'] <= 4)]
        df_filtered = df_filtered[(df_filtered['Jumlah Lantai'] != 0) & (df_filtered['Jumlah Lantai'] != 4)]
        df_filtered = df_filtered[(df_filtered['Sertifikat'] != "Strata") & (df_filtered['Sertifikat'] != 'Hak Sewa')]
        df_filtered = df_filtered[(df_filtered['Luas Bangunan'] <= 1000) & (df_filtered['Luas Tanah'] <= 1000)]
        
        return df_filtered
    
    @staticmethod
    def encode_categorical_features(df):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        # Sertifikat encoding
        sertif_enc = OrdinalEncoder(categories=[ENCODING_CATEGORIES['sertifikat']])
        df_encoded[['Sertifikat']] = sertif_enc.fit_transform(df_encoded[['Sertifikat']])
        
        # Kondisi encoding
        kondisi_enc = OrdinalEncoder(categories=[ENCODING_CATEGORIES['kondisi']])
        df_encoded[['Kondisi Properti']] = kondisi_enc.fit_transform(df_encoded[['Kondisi Properti']])
        
        # Daya Listrik encoding
        watt_enc = OrdinalEncoder(categories=[ENCODING_CATEGORIES['daya_listrik']])
        df_encoded[['Daya Listrik']] = watt_enc.fit_transform(df_encoded[['Daya Listrik']])
        
        # Kecamatan one-hot encoding
        df_encoded = pd.get_dummies(df_encoded, columns=['Kecamatan'], prefix='kec')
        
        return df_encoded
    
    @staticmethod
    def apply_log_transformation(df, columns=None):
        """Apply log transformation to specified columns"""
        if columns is None:
            columns = ['Harga', 'Luas Tanah', 'Luas Bangunan']
        
        df_transformed = df.copy()
        for col in columns:
            if col in df_transformed.columns:
                df_transformed[col] = np.log1p(df_transformed[col])
        
        return df_transformed


class Visualizer:
    """Class for creating visualizations"""
    
    @staticmethod
    def plot_location_distribution(df, kecamatan_col='Kecamatan', figsize=(10, 6)):
        """Plot distribution of house locations"""
        plt.figure(figsize=figsize)
        ax = sns.countplot(y=df[kecamatan_col], order=df[kecamatan_col].value_counts().index)
        plt.title("House Location Distribution", weight='bold')
        
        # Add labels to bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_width())}', 
                       (p.get_width(), p.get_y() + p.get_height() / 2),
                       ha='left', va='center', fontsize=10, fontweight='bold', 
                       xytext=(5, 0), textcoords='offset points')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_categorical_distributions(df, figsize=(12, 10)):
        """Plot distributions of categorical features"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        columns = ['Jumlah Lantai', 'Carport', 'Sertifikat', 'Kondisi Properti']
        titles = ['Floor Count Distribution', 'Carport Distribution', 
                 'Certificate Distribution', 'Property Condition Distribution']
        
        for i, ax in enumerate(axes.flat):
            sns.countplot(x=df[columns[i]], ax=ax)
            ax.set_title(titles[i], weight='bold')
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                           (p.get_x() + p.get_width() / 2, p.get_height()), 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_numeric_vs_price(df, numerical_vars, target_col='Harga', figsize=(20, 15)):
        """Plot scatter plots of numeric variables vs price"""
        num_vars = len(numerical_vars)
        num_cols = 4
        num_rows = -(-num_vars // num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i, num_var in enumerate(numerical_vars):
            sns.scatterplot(x=df[num_var], y=df[target_col], ax=axes[i], alpha=0.5)
            axes[i].set_title(f'Price vs {num_var}')
            axes[i].set_xlabel(num_var)
            axes[i].set_ylabel('Price')
        
        # Remove empty subplots
        for j in range(len(numerical_vars), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_all_kecamatan_boxplots(df, harga_col='Harga', figsize=(15, 20)):
        """
        Create boxplots for all kecamatan one-hot encoded columns showing price distribution
        and percentage difference from overall average
        """
        # Select all columns containing 'kec_' (one-hot encoding results)
        categorical_vars = [col for col in df.columns if col.startswith('kec_')]
        
        # Calculate the overall average price
        avg_price_overall = df[harga_col].mean()
        
        # Determine the number of subplots based on the number of subdistricts
        num_vars = len(categorical_vars)
        num_cols = 4
        num_rows = -(-num_vars // num_cols)  # Ceil division
        
        # Create subplot grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        # Create a boxplot for each subdistrict
        for i, cat_var in enumerate(categorical_vars):
            sns.boxplot(x=df[cat_var], y=df[harga_col], ax=axes[i])
            
            # Calculate the average price for the subdistrict (only if 'Yes' or 1)
            avg_price_kecamatan = df[df[cat_var] == 1][harga_col].mean()
            
            # Calculate the percentage difference
            pct_diff = ((avg_price_kecamatan - avg_price_overall) / avg_price_overall) * 100
            
            # Update the title with the percentage
            title = f"House Prices in {cat_var.replace('kec_', '')}\n({pct_diff:.2f}% from average)"
            axes[i].set_title(title)
            
            # Set the X-axis label
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(['No', 'Yes'])
            axes[i].set_xlabel('')
        
        # Delete empty subplots if the number of subdistricts is less than the number of grids
        for j in range(len(categorical_vars), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
        
        # Print overall average price
        print(f"Overall average price: {avg_price_overall:,.2f}")
        
        return avg_price_overall
    
    @staticmethod
    def plot_kecamatan_price_comparison(df_kabupaten, top_n=2):
        """Plot price comparison for top and bottom districts"""
        categorical_vars = [col for col in df_kabupaten.columns if col.startswith('kec_')]
        avg_price_overall = df_kabupaten['Harga'].mean()
        
        # Calculate average prices by district
        kecamatan_data = []
        for cat_var in categorical_vars:
            kec_name = cat_var.replace('kec_', '')
            avg_price = df_kabupaten[df_kabupaten[cat_var] == 1]['Harga'].mean()
            pct_diff = ((avg_price - avg_price_overall) / avg_price_overall) * 100
            kecamatan_data.append({'kecamatan': kec_name, 'avg_price': avg_price, 'pct_diff': pct_diff})
        
        df_kecamatan = pd.DataFrame(kecamatan_data)
        df_kecamatan_sorted = df_kecamatan.sort_values('pct_diff', ascending=False)
        
        # Select top and bottom districts
        top_n_data = df_kecamatan_sorted.head(top_n)
        bottom_n_data = df_kecamatan_sorted.tail(top_n)
        selected_kecamatan = pd.concat([top_n_data, bottom_n_data])
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (idx, row) in enumerate(selected_kecamatan.iterrows()):
            cat_var = 'kec_' + row['kecamatan']
            sns.boxplot(x=df_kabupaten[cat_var], y=df_kabupaten['Harga'], ax=axes[i])
            
            # Format title with colors
            title_color = 'green' if row['pct_diff'] > 0 else 'red'
            direction = "↑" if row['pct_diff'] > 0 else "↓"
            
            title = f"{row['kecamatan']}\n{direction} {abs(row['pct_diff']):.2f}% from average"
            axes[i].set_title(title, color=title_color, fontweight='bold')
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(['No', 'Yes'])
            axes[i].set_ylabel('Price' if i % 2 == 0 else '')
            axes[i].set_xlabel('')
        
        plt.suptitle(f'House Price Comparison in Top {top_n} and Bottom {top_n} Districts', 
                    y=1.02, fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Format the return table with proper currency and percentage formatting
        formatted_kecamatan = selected_kecamatan.copy()
        formatted_kecamatan['avg_price'] = formatted_kecamatan['avg_price'].apply(
            lambda x: f"Rp {x:,.0f}" if not pd.isna(x) else "N/A"
        )
        formatted_kecamatan['pct_diff'] = formatted_kecamatan['pct_diff'].apply(
            lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A"
        )
        
        # Sort by percentage difference for display
        formatted_kecamatan = formatted_kecamatan.sort_values('pct_diff', ascending=False)
        
        return formatted_kecamatan
    
    @staticmethod
    def plot_price_distributions(df, figsize=(15, 5)):
        """Plot distributions of price, land area, and building area"""
        plt.figure(figsize=figsize)
        
        # Price
        plt.subplot(1, 3, 1)
        sns.histplot(df['Harga'], kde=True, color='blue')
        plt.title('House Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        
        # Land area
        plt.subplot(1, 3, 2)
        sns.histplot(df['Luas Tanah'], kde=True, color='green')
        plt.title('Land Area Distribution')
        plt.xlabel('Land Area (m²)')
        plt.ylabel('Frequency')
        
        # Building area
        plt.subplot(1, 3, 3)
        sns.histplot(df['Luas Bangunan'], kde=True, color='orange')
        plt.title('Building Area Distribution')
        plt.xlabel('Building Area (m²)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_heatmap(df, figsize=(12, 10)):
        """Plot correlation heatmap for numeric columns"""
        plt.figure(figsize=figsize)
        sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_scatter_grids(df, target_col='Harga', threshold=0.3, base_figsize=(20, 5)):
        """
        Create scatter plot grids for variables with strong and weak correlations
        with dynamic figure sizing based on number of variables
        """
        # Select all numeric columns except the target
        excluded_columns = [target_col]
        numerical_vars = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                         if col not in excluded_columns]
        
        # Calculate correlation with target
        correlations = df[numerical_vars].corrwith(df[target_col]).abs().sort_values(ascending=False)
        
        # Separate variables based on correlation strength
        strong_corr_vars = correlations[correlations > threshold].index.tolist()
        weak_corr_vars = correlations[correlations <= threshold].index.tolist()
        
        # Plot variables with strong correlations
        if strong_corr_vars:
            Visualizer._create_scatter_grid(df, strong_corr_vars, target_col, correlations, 
                                          'Relationship Between Price and Predictor Variables (Strong Correlation)',
                                          base_figsize)
        
        # Plot variables with weak correlations
        if weak_corr_vars:
            Visualizer._create_scatter_grid(df, weak_corr_vars, target_col, correlations,
                                          'Relationship Between Price and Predictor Variables (Weak Correlation)',
                                          base_figsize)
        
        # Return correlation analysis results
        return {
            'correlations': correlations,
            'strong_correlation_vars': strong_corr_vars,
            'weak_correlation_vars': weak_corr_vars
        }
    
    @staticmethod
    def _create_scatter_grid(df, variables, target_col, correlations, title, base_figsize=(20, 5)):
        """Helper method to create scatter plot grid with dynamic sizing"""
        num_vars = len(variables)
        if num_vars == 0:
            return
        
        num_cols = 4
        num_rows = -(-num_vars // num_cols)  # Ceil division
        
        # Dynamic figure sizing based on number of rows
        fig_width = base_figsize[0]
        fig_height = base_figsize[1] * num_rows  # Scale height with number of rows
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten()
        
        for i, var in enumerate(variables):
            sns.scatterplot(x=df[var], y=df[target_col], ax=axes[i], alpha=0.5)
            corr_value = correlations[var]
            axes[i].set_title(f'Price vs {var}\n(corr: {corr_value:.2f})', fontsize=12)
            axes[i].set_xlabel(var, fontsize=10)
            axes[i].set_ylabel('Price', fontsize=10)
        
        # Delete empty subplots
        for j in range(len(variables), len(axes)):
            fig.delaxes(axes[j])
        
        plt.suptitle(title, y=0.98, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


class DataAnalyzer:
    """Class for data analysis utilities"""
    
    @staticmethod
    def calculate_skewness(df, columns=None):
        """Calculate skewness for specified columns and print formatted results"""
        if columns is None:
            columns = ['Harga', 'Luas Tanah', 'Luas Bangunan']
        
        skewness = {}
        for col in columns:
            if col in df.columns:
                skew_val = np.round(df[col].skew(), 2)
                skewness[col] = skew_val
                # Print formatted output
                print(f"Skewness of the '{col}' column distribution: {skew_val:>5}")
        
        return skewness
    
    @staticmethod
    def analyze_correlations(df, target_col='Harga', threshold=0.3):
        """Analyze correlations with target variable"""
        excluded_columns = [target_col]
        numerical_vars = [col for col in df.select_dtypes(include=['number']).columns 
                         if col not in excluded_columns]
        
        correlations = df[numerical_vars].corrwith(df[target_col]).abs().sort_values(ascending=False)
        
        strong_corr_vars = correlations[correlations > threshold].index.tolist()
        weak_corr_vars = correlations[correlations <= threshold].index.tolist()
        
        return {
            'correlations': correlations,
            'strong_correlation_vars': strong_corr_vars,
            'weak_correlation_vars': weak_corr_vars
        }

class ModelEvaluator:
    """Simplified model evaluation class following Gemini's approach"""
    
    @staticmethod
    def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
        """
        Performs 5-fold cross-validation, fits a model, and evaluates it on train and test sets.
        Returns metrics dictionary with predictions for all sets.
        """
        # Define metrics for cross-validation
        scoring = {
            'r2': 'r2', 
            'rmse': 'neg_root_mean_squared_error', 
            'mape': 'neg_mean_absolute_percentage_error'
        }
        
        # Perform cross-validation with estimators to get validation predictions
        cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring, 
                                  n_jobs=-1, return_estimator=True)
        
        # Get cross-validation predictions for validation set
        cv_predictions = cross_val_predict(model, X_train, y_train, cv=5, n_jobs=-1)
        
        # Fit the model on the entire training data
        model.fit(X_train, y_train)
        
        # Make predictions on train and test sets
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Store all metrics in a dictionary
        metrics = {
            'Model': model_name,
            'Train_R2': r2_score(y_train, y_pred_train),
            'Validation_R2': cv_results['test_r2'].mean(),
            'Test_R2': r2_score(y_test, y_pred_test),
            'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'Validation_RMSE': -cv_results['test_rmse'].mean(),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Train_MAPE': mean_absolute_percentage_error(y_train, y_pred_train),
            'Validation_MAPE': -cv_results['test_mape'].mean(),
            'Test_MAPE': mean_absolute_percentage_error(y_test, y_pred_test),
            'fitted_model': model,
            'predictions': {
                'train': y_pred_train,
                'validation': cv_predictions,
                'test': y_pred_test
            }
        }
        
        return metrics
    
    @staticmethod
    def tune_with_optuna(model_name, X_train, y_train, n_trials=50):
        """
        Tunes hyperparameters for either Random Forest or XGBoost using Optuna.
        Returns the best parameters and study object.
        """
        def objective(trial):
            if model_name == 'Random Forest':
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 1200, 2000, step=100),
                    "max_depth": trial.suggest_int("max_depth", 20, 50, step=5),
                    "min_samples_split": trial.suggest_int("min_samples_split", 4, 12),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                    "max_features": trial.suggest_float("max_features", 0.4, 0.7),
                    "max_samples": trial.suggest_float("max_samples", 0.8, 0.95),
                    "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.005),
                }
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            
            elif model_name == 'XGBoost':
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 1200, 2500, step=100),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, log=True),
                    "max_depth": trial.suggest_int("max_depth", 4, 7),
                    "min_child_weight": trial.suggest_int("min_child_weight", 2, 6),
                    "subsample": trial.suggest_float("subsample", 0.85, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 0.95),
                    "gamma": trial.suggest_float("gamma", 0.1, 3),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1, 5)
                }
                model = XGBRegressor(**params, random_state=42, n_jobs=-1)
            else:
                raise ValueError("Model name not supported.")

            # Use R² score for optimization
            return cross_validate(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)['test_score'].mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best hyperparameters for {model_name}: {study.best_params}")
        
        return study.best_params, study
    
    @staticmethod
    def create_results_table(metrics_list):
        """Create a beautiful results table with grouped Train/Validation/Test"""
        # Create a new list to store reformatted data
        reformatted_data = []
        
        for metrics in metrics_list:
            model_name = metrics['Model']
            
            # Add all datasets for this model
            reformatted_data.append({
                'Model': model_name,
                'Dataset': 'Train',
                'R2': metrics['Train_R2'],
                'RMSE': metrics['Train_RMSE'],
                'MAPE': metrics['Train_MAPE']
            })
            
            reformatted_data.append({
                'Model': model_name,
                'Dataset': 'Validation',
                'R2': metrics['Validation_R2'],
                'RMSE': metrics['Validation_RMSE'],
                'MAPE': metrics['Validation_MAPE']
            })
            
            reformatted_data.append({
                'Model': model_name,
                'Dataset': 'Test',
                'R2': metrics['Test_R2'],
                'RMSE': metrics['Test_RMSE'],
                'MAPE': metrics['Test_MAPE']
            })
        
        df = pd.DataFrame(reformatted_data)
        
        # Apply conditional formatting for better visual hierarchy
        def highlight_dataset(val):
            if val == 'Train':
                return 'background-color: #e8f5e8'  # Light green
            elif val == 'Validation':
                return 'background-color: #e3f2fd'  # Light blue  
            elif val == 'Test':
                return 'background-color: #fff3e0'  # Light orange
            return ''
        
        styled_df = df.style.format({
            'R2': '{:.4f}',
            'RMSE': '{:.4f}',
            'MAPE': '{:.4f}'
        }).applymap(highlight_dataset, subset=['Dataset'])\
        .set_properties(**{
            'border': '1px solid #dee2e6'
        }).set_table_styles([
            {
                'selector': 'th',
                'props': [('background-color', '#343a40'), 
                        ('color', 'white'),
                        ('font-weight', 'bold'),
                        ('text-align', 'center')]
            },
            {
                'selector': 'td',
                'props': [('text-align', 'center')]
            },
            {
                'selector': 'td:first-child',
                'props': [('text-align', 'left', 'font-weight', 'bold')]
            }
        ])
        
        return df, styled_df
    
    @staticmethod
    def plot_convergence(study, model_name):
        """Plot convergence plot for Optuna study"""
        trials = study.trials_dataframe()
        
        plt.figure(figsize=(10, 5))
        plt.plot(trials["number"], trials["value"], marker="o", linestyle="-", 
                label=f"{model_name} R²", alpha=0.7)
        plt.axhline(y=max(trials["value"]), color="r", linestyle="--", 
                   label=f"Best Score: {max(trials['value']):.4f}")
        
        plt.xlabel("Number of Trials")
        plt.ylabel("R² Score")
        plt.title(f"Convergence Plot - {model_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_learning_curve(model, X, y, model_name, cv_splits=5):
        """Plot learning curve for a model"""
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv_splits, scoring="r2", n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_sizes, train_mean, label="Train R²", marker="o", color="blue", linewidth=2)
        plt.plot(train_sizes, test_mean, label="Validation R²", marker="o", color="green", linewidth=2)
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        color="blue", alpha=0.2)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                        color="green", alpha=0.2)
        
        plt.xlabel("Training Set Size")
        plt.ylabel("R² Score")
        plt.title(f"Learning Curve - {model_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_predictions_comparison(metrics_list, y_train, y_test, set_type='test'):
        """
        Plot actual vs predicted values for all models for a specific set
        """
        n_models = len(metrics_list)
        n_cols = 2
        n_rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_models > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, metrics in enumerate(metrics_list):
            model_name = metrics['Model']
            
            if set_type == 'train':
                y_actual = y_train
                y_pred = metrics['predictions']['train']
                title_suffix = 'Train'
            elif set_type == 'validation':
                y_actual = y_train
                y_pred = metrics['predictions']['validation']
                title_suffix = 'Validation'
            else:  # test
                y_actual = y_test
                y_pred = metrics['predictions']['test']
                title_suffix = 'Test'
            
            # Convert back from log if needed
            if y_actual.max() < 100:  # Assuming log transformation
                y_actual_plot = np.expm1(y_actual)
                y_pred_plot = np.expm1(y_pred)
            else:
                y_actual_plot = y_actual
                y_pred_plot = y_pred
            
            sns.regplot(x=y_actual_plot, y=y_pred_plot, scatter_kws={"alpha": 0.5}, 
                       line_kws={"color": "red"}, ax=axes[i])
            axes[i].set_title(f'{model_name} - {title_suffix} Set')
            axes[i].set_xlabel('Actual Price')
            axes[i].set_ylabel('Predicted Price')
            
            # Add R² to plot
            r2 = r2_score(y_actual_plot, y_pred_plot)
            axes[i].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[i].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Remove empty subplots
        for j in range(len(metrics_list), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_prediction_samples(metrics_list, y_train, y_test, n_samples=5):
        """
        Create sample prediction comparisons for all models and sets
        """
        samples = {}
        
        for metrics in metrics_list:
            model_name = metrics['Model']
            
            # Convert back from log if needed
            if y_train.max() < 100:
                y_train_actual = np.expm1(y_train)
                y_test_actual = np.expm1(y_test)
                train_pred = np.expm1(metrics['predictions']['train'])
                val_pred = np.expm1(metrics['predictions']['validation'])
                test_pred = np.expm1(metrics['predictions']['test'])
            else:
                y_train_actual = y_train
                y_test_actual = y_test
                train_pred = metrics['predictions']['train']
                val_pred = metrics['predictions']['validation']
                test_pred = metrics['predictions']['test']
            
            # Create sample DataFrames
            train_sample = pd.DataFrame({
                'Actual_Train': y_train_actual[:n_samples],
                f'Predicted_{model_name}': train_pred[:n_samples],
                f'Difference_{model_name}': train_pred[:n_samples] - y_train_actual[:n_samples]
            })
            
            val_sample = pd.DataFrame({
                'Actual_Validation': y_train_actual[:n_samples],
                f'Predicted_{model_name}': val_pred[:n_samples],
                f'Difference_{model_name}': val_pred[:n_samples] - y_train_actual[:n_samples]
            })
            
            test_sample = pd.DataFrame({
                'Actual_Test': y_test_actual[:n_samples],
                f'Predicted_{model_name}': test_pred[:n_samples],
                f'Difference_{model_name}': test_pred[:n_samples] - y_test_actual[:n_samples]
            })
            
            samples[model_name] = {
                'train': train_sample,
                'validation': val_sample,
                'test': test_sample
            }
        
        return samples