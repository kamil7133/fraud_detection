# Credit Card Fraud Detection Project

This repository contains a comprehensive machine learning pipeline for detecting credit card fraud. The project follows a structured approach from data exploration to model deployment with a minimal Streamlit interface.

Used Dataset
- https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud
## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
  - [Running Notebooks](#running-notebooks)
  - [Launching the Streamlit App](#launching-the-streamlit-app)
- [Results](#results)
- [License](#license)
- [Screenshots](#screenshots)

## Project Overview
The objective of this project is to detect fraudulent transactions using a supervised machine learning approach. The workflow includes:
- **Exploratory Data Analysis (EDA):** Understanding the dataset, handling missing values, checking class balance.
- **Preprocessing:** Data cleaning, balancing (using SMOTE), feature scaling.
- **Model Training:** Building and evaluating multiple models (Logistic Regression, Random Forest, XGBoost, Neural Network).
- **Model Tuning:** Optimizing the best-performing model using GridSearchCV.
- **Final Evaluation:** Testing the final model on an unseen test set.
- **Deployment (Optional):** A simple Streamlit app for interactive fraud detection demo.

## Data
The dataset used is a credit card transactions dataset that contains various features about each transaction and a target column indicating whether the transaction was fraudulent (`fraud = 1`) or normal (`fraud = 0`). Data is stored in the `data/raw/` directory and is processed during the preprocessing stage.

## Project Structure
```plaintext
fraud_detection/
├── data/
│   ├── raw/                 # Raw dataset files (e.g., CSV from Kaggle)
│   └── processed/           # Processed data files (train.pkl, val.pkl, test.pkl)
├── notebooks/               
│   ├── 01_exploration.ipynb # EDA and initial analysis
│   ├── 02_preprocessing.ipynb  # Data cleaning, balancing, scaling, saving datasets and scaler
│   ├── 03_model_training.ipynb # Training baseline models, model comparison
│   ├── 04_model_tuning.ipynb   # (Optional) Hyperparameter tuning and advanced improvements
│   └── 05_evaluation.ipynb     # Final model evaluation on test set, reporting    
├── app/
│   └── streamlit_app.py     # Streamlit application for model demonstration
├── models/
│   ├── best_random_forest.pkl # Exported trained model
│   └── scaler.pkl             # Exported scaler for feature normalization
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

- **data/**: Contains raw and processed data.
- **notebooks/**: Jupyter notebooks demonstrating each stage of the pipeline.
- **app/**: Streamlit application for interactive fraud detection.
- **models/**: Contains saved models and scaler objects.
- **requirements.txt**: List of dependencies.
- **README.md**: This documentation file.

## Installation and Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Notebooks
Open the Jupyter notebooks to follow each step of the pipeline:
```bash
jupyter lab  # or jupyter notebook
```
Then, open the notebooks in the `notebooks/` directory sequentially:
1. `01_exploration.ipynb`
2. `02_preprocessing.ipynb`
3. `03_model_training.ipynb`
4. (Optional) `04_model_tuning.ipynb`
5. `05_evaluation.ipynb`

### Launching the Streamlit App
To run the interactive fraud detection demo:
1. Navigate to the project root directory.
2. Launch Streamlit:
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. The application will open in your web browser at `http://localhost:8501/`. You can input transaction details to see if they are predicted as fraudulent or normal.

## Results
The Random Forest model, fine-tuned using GridSearchCV (with parameters `n_estimators=200, max_depth=None`), achieved near-perfect performance on the validation set. The final evaluation on the test set confirmed the model's high accuracy. 

The Streamlit app allows you to input new transaction data and get real-time fraud detection predictions based on the trained model.

## License
This project is licensed under the MIT License.

## Demo
You can test it by yourself under this link:


## Screenshots
*(Add screenshots or GIFs demonstrating the Streamlit app, EDA plots, model evaluation metrics, etc.)*

---

*Feel free to explore the notebooks and adjust the parameters as needed. Contributions and suggestions are welcome!*