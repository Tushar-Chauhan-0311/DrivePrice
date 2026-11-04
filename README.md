# DrivePrice – Used Car Price Prediction

DrivePrice is a machine learning-based web application that predicts the estimated selling price of a used car based on its specifications, such as manufacturing year, showroom price, kilometers driven, fuel type, seller type, transmission, and number of previous owners.

The application uses a trained Random Forest regression model and provides an interactive interface built with Streamlit.

## Features

- Predicts the estimated selling price of a used car.
- Takes multiple vehicle details as input.
- Performs data preprocessing and feature engineering.
- Handles categorical variables using One-Hot Encoding.
- Compares Linear Regression, Decision Tree, and Random Forest models.
- Evaluates models using MAE, RMSE, and R².
- Provides an interactive web interface using Streamlit.

## Tech Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Data Visualization:** Matplotlib
- **Web Framework:** Streamlit

## Machine Learning Workflow

1. **Data Collection:** Loaded the used car dataset containing vehicle specifications and selling prices.
2. **Data Cleaning:** Checked missing values and removed duplicate records.
3. **Feature Engineering:** Created a `Car_Age` feature using the reference year 2018.
4. **Preprocessing:** Applied One-Hot Encoding to categorical features.
5. **Model Training:** Trained Linear Regression, Decision Tree, and Random Forest models.
6. **Model Evaluation:** Compared models using repeated 5-fold cross-validation and test-set evaluation.
7. **Model Selection:** Selected Random Forest based on its cross-validation MAE.
8. **Deployment:** Integrated the trained model into a Streamlit application.

## Model Performance

The following are the results from the current training run:

| Metric | Random Forest |
|---|---:|
| Cross-validation MAE | 0.717 lakh |
| Cross-validation R² | 0.892 |
| Test MAE | 1.375 lakh |
| Test RMSE | 3.240 lakh |
| Test R² | 0.593 |

The cross-validation and test results differ, indicating that the model's performance varies across data splits. The test results should be considered when judging its ability to generalize to unseen cars.

## Project Structure

```text
DrivePrice/
│
├── app.py
├── train_model.py
├── prepare_data.py
├── split_data.py
├── inspect_data.py
├── explore_data.py
├── check_dataset.py
│
├── data/
│   └── used_cars.csv
│
├── car_price_model.pkl
├── requirements.txt
└── README.md
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DrivePrice
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the application**

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Limitations

- The model is trained on a relatively small historical dataset.
- The dataset contains vehicle records from 2003 to 2018.
- The reference year used for calculating car age is 2018.
- Predictions are estimates and may not reflect current market prices.
- Model performance may vary for vehicles with limited representation in the dataset.

## Future Improvements

- Train the model on a larger and more recent dataset.
- Improve prediction performance for different vehicle categories.
- Add actual-versus-predicted price visualizations.
- Deploy the application online.
- Explore additional machine learning algorithms and feature engineering techniques.

---

**Disclaimer:** DrivePrice is an educational machine learning project. Its predictions are estimates and should not be treated as actual market valuations.
