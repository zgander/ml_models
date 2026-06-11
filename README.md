# Housing Price Predictor using Linear Regression

## Project Overview
This repository presents an end-to-end Machine Learning project to predict housing prices using supervised Linear Regression. The project covers the entire workflow from initial data loading, cleaning, and exploratory data analysis (EDA) to feature engineering, model training, and performance evaluation. Using a dataset of housing characteristics, we establish a robust pipeline using standard Python libraries to train a model that evaluates the relationships between home attributes and sale prices.

## Problem Statement
In the real estate sector, manual property valuation is slow and subject to bias. This project addresses the challenge by developing an interpretable, regression-based model to predict property sale prices based on physical attributes and qualitative conditions. Using Linear Regression, we construct a mathematical model that maps these attributes to a continuous price target, providing transparent coefficient weights to understand which features drive housing values.

## Dataset Information
The model is trained on the **Housing Prices Dataset** by Yasser H. (Kaggle).
* **Target Variable:** `price` (continuous house sale price)
* **Dataset Size:** 545 instances
* **Initial Feature Set:** 13 variables
* **Dataset Attributes:** The dataset contains continuous metrics (such as `area`), discrete properties (such as counts of `bedrooms`, `bathrooms`, `stories`, and `parking`), and nominal variables describing amenities and location status.

## Data Preprocessing
A concise data preparation pipeline was implemented to clean and structure the features for linear modeling:
* **Cleaning:** Checked for missing values and duplicate records. The dataset had zero missing values and zero duplicates.
* **Encoding:** Binary variables containing "yes"/"no" states (e.g., `mainroad`, `guestroom`, `basement`, `prefarea`) were encoded to 1/0.
* **One-Hot Encoding:** The categorical feature `furnishingstatus` was converted using one-hot encoding, dropping the baseline category to avoid multicollinearity.
* **Feature Distribution:** Analyzed the distribution of each feature with histplots to understand its characteristics, and skewness
* **Outliers:** Evaluated using boxplots. Outliers in `price` and `area` were retained as they represent valid high-value properties.
* **Correlation Analysis:** Analysed the degree and nature of relation between various features and `price` using a correlation heatmap and Variable Inflation Factor to identify and eliminate multicollinearity.
* **Feature Removal:** Removed `semi-furnishedstatus` column from the predictor set due to high linear dependency with other features and low correlation with `price`

## Project Workflow
The sequential structure of the project pipeline is shown below:

```
Dataset
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Feature Engineering & Encoding
   ↓
Correlation Analysis
   ↓
Train-Test Split
   ↓
Linear Regression Model
   ↓
Predictions
   ↓
Model Evaluation
```

## Model Development
Model training followed a standard supervised learning protocol:
* **Feature-Target Separation:** Separated predictors ($X$) from the target variable ($y = \text{price}$).
* **Train-Test Split:** Partitioned the data into an 80% training set and a 20% testing set using a fixed random state to ensure reproducibility.
* **Implementation:** Fit an Ordinary Least Squares (OLS) Linear Regression model using Scikit-Learn.
* **Prediction:** Generated predictions on the unseen testing partition to evaluate performance.

## Model Performance
The performance of the Linear Regression model was evaluated using standard regression metrics. The results are detailed below:

* **Mean Absolute Error (MAE):** `970043.4039201641`
* **Mean Squared Error (MSE):** `1754318687330.6646`
* **Training $R^2$ Score:** `0.6859438988560158`
* **Testing $R^2$ Score:** `0.6529242642153182`

### Interpretation of Results
The model achieved a training $R^2$ of approximately 0.68 and a testing $R^2$ of 0.65. This indicates that the selected features explain approximately 65% of the variance in housing prices on unseen test data. The minimal difference between training and testing performance confirms that the model generalizes well and does not suffer from overfitting.

## Project Structure
The repository is organized as follows:
```
HousingLinearRegression/
├── housing_cleaned.csv            # Preprocessed dataset
├── model.py                       # Model training and evaluation script
├── README.md                      # Project documentation
└── SmallHousingPrice.ipynb        # Data preprocessing and EDA notebook
```

## Installation and Setup
To set up this project locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/username/housing-linear-regression.git
   cd housing-linear-regression
   ```

2. **Install Dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
   
3. **Usage Instructions**
To run the model training and evaluation script:
1. Ensure `housing_cleaned.csv` is in the same directory as `model.py`.
2. Run the execution script:
   ```bash
   python model.py
   ```

## Technologies Used
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn

## Conclusion
This project demonstrates the application of Linear Regression to predict real estate pricing. By executing a clean pipeline of preprocessing, categorical encoding, and feature selection, we established a structured dataset for machine learning. The final model explains approximately 65% of the variation in housing prices. The remaining 35% variation may be attributed to factors not present in the dataset, non-linear relationships between features and price, and inherent noise commonly found in real-world housing markets.

## Author
* **Name:** Preetham Vasireddy
* **Email:** preethamdeoxys@gmail.com
* **GitHub:** @zgander(https://github.com/zgander)
* **LinkedIn:** Preetham Vasireddy(https://www.linkedin.com/in/preetham-vasireddy-967924327/)
