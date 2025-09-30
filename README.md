# 🏡 Tangerang House Price Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains the source code for my final thesis project: an end-to-end machine learning pipeline for predicting house prices in Tangerang Regency, Indonesia. The project handles the entire data lifecycle, from automated web scraping and data preprocessing to advanced model tuning, evaluation, and deployment as an interactive web application.

---

## ✨ Key Features

* **End-to-End Data Pipeline:** A complete workflow that starts with automated web scraping and ends with a deployed, interactive predictive model.
* **Advanced Regression Modeling:** Compares multiple regression models, including Linear Regression, Random Forest, and a final **XGBoost** model tuned with **Optuna** for optimal performance.
* **Interactive Application:** A user-friendly web app built with **Streamlit** that allows users to input house features and receive an instant price estimate.
* **Containerized for Reproducibility:** The entire application is containerized using **Docker**, ensuring a consistent and easy setup for any user on any machine.

---

## 🏗️ Project Workflow

The project is structured as a sequential data pipeline:

1.  **Web Scraping (`src/scraping/`):** A two-part scraper first crawls `Rumah123.com` using **Selenium** to gather property URLs, then scrapes the details from each URL using **BeautifulSoup** and **Requests**.
2.  **Data Preprocessing (`helpers.py`):** A robust preprocessing function handles data cleaning, type conversion, feature engineering (e.g., extracting `Kecamatan`), outlier handling, and encoding.
3.  **Modeling (`house-price-prediction.ipynb`):** The notebook covers exploratory data analysis (EDA), model training, and hyperparameter tuning with Optuna to find the best-performing XGBoost model.
4.  **Deployment (`src/app/app.py`):** The final trained model, scaler, and encoders are served via a user-friendly Streamlit application, which is then containerized with Docker.

---

## 📈 Model Performance & Key Findings

The final **XGBoost** model, tuned with Optuna, was the top performer.

| Metric | Test Set Performance |
| :--- | :--- |
| **R-squared (R²)** | `0.9044`  |
| **RMSE (in Rupiah)** | `Rp 1,188,284,228` |

The most influential features in predicting house prices were identified as:
* `Luas Tanah` (Land Area)
* `Luas Bangunan` (Building Area)
* The property's location (`Kecamatan`)

---

## 🛠️ Tech Stack

* **Data Scraping:** Selenium, BeautifulSoup, Requests
* **Data Analysis:** Python, Pandas, NumPy, Scikit-learn
* **Modeling:** XGBoost, Optuna
* **App & Deployment:** Streamlit, Docker
* **Visualization:** Matplotlib, Seaborn

---

## 🚀 Setup and Installation

### Method 1: Running with Docker (Recommended)
This is the simplest way to run the application.

1.  **Build the Docker image:**
    ```bash
    docker build -t house-price-app .
    ```
2.  **Run the container:**
    ```bash
    docker run -p 8501:8501 house-price-app
    ```
3.  Open your web browser and navigate to `http://localhost:8501`.

### Method 2: Running Locally (Manual Setup)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/daff00/tangerang-house-price-pipeline.git](https://github.com/daff00/tangerang-house-price-pipeline.git)
    ```
2.  **Create and activate a Python virtual environment:**
    > This project was developed and tested using **Python 3.9**.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Replace path on `# Load Ordinal Encoder`, `# Load scaler`, `# Load model` on `app.py`:**
    ```bash
    #from
    #Load Ordinal Encoder
        with open("models/encoder_data_listrik.pkl", "rb") as f:
            watt_enc = pickle.load(f)
        
    # Load scaler
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    # Load model
        with open("models/xgboost_optuna.pkl", "rb") as f:
            model_xgb = pickle.load(f)

    # to
    #Load Ordinal Encoder
        with open("../../models/encoder_data_listrik.pkl", "rb") as f:
            watt_enc = pickle.load(f)
        
    # Load scaler
        with open("../../models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    # Load model
        with open("../../models/xgboost_optuna.pkl", "rb") as f:
            model_xgb = pickle.load(f)
    ```
5.  **Run the Streamlit App:**
    ```bash
    streamlit run src/app/app.py
    ```

---

## 📂 Project Structure
<details>
<summary>Click to view the project directory structure</summary>
tangerang-house-price-pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── app/
│   │   └── app.py
│   └── scraping/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
</details>

---

## 📬 Contact

* **Daffa Kaisha Pratama Chandra** - [daffakpc21@gmail.com](mailto:your.email@daffakpc21@gmail.com)
* **LinkedIn:** [https://www.linkedin.com/in/daffakaisha/](https://www.linkedin.com/in/daffakaisha/)