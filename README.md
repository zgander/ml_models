# Housing Price Predictor using Linear Regression

## Project Overview

This project implements a rigorous data preparation workflow on the Kaggle Housing Prices dataset. By transforming raw, unstructured columns into clean, scaled, and non-multicollinear numerical features, we establish a robust foundation for building an interpretable Linear Regression model in Phase 2. 

This repository documents the entire preprocessing lifecycle, from raw data ingestion to the export of a ready-to-train dataset.

---

## Problem Statement

Predicting residential property prices is a classic regression challenge. Housing prices are influenced by a mixture of continuous variables (e.g., floor area), ordinal scales (e.g., stories, bedrooms), and nominal conditions (e.g., furnishing status, presence of air conditioning). 

To train an optimal Linear Regression model, the dataset must satisfy specific linear assumptions:
* Categorical features must be mapped to appropriate numerical representations.
* Multi-category columns must not introduce perfect multicollinearity (the "dummy variable trap").
* Predictor variables must not display severe collinearity, which inflates the variance of coefficient estimates.
* Data errors must be purged, while legitimate statistical outliers must be evaluated to ensure the model remains generalizable.

This phase systematically addresses these challenges to deliver a clean dataset.

---

## Dataset Information

* **Source:** Housing Prices Dataset by Yasser H. (available on Kaggle)
* **Instance Count:** 545 records
* **Feature Count:** 13 columns (12 predictors, 1 target variable)
* **Target Variable:** `price` (continuous numerical value representing the property sale price)
* **Programming Language:** Python 3.13

---

## Features Used

Below is the list of raw features analyzed during this phase:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `price` | Integer | Sale price of the house (Target Variable). |
| `area` | Integer | Total area of the house in square feet. |
| `bedrooms` | Integer | Number of bedrooms in the house. |
| `bathrooms` | Integer | Number of bathrooms in the house. |
| `stories` | Integer | Number of storeys/floors in the building structure. |
| `mainroad` | Object | Is the property connected to a main road? (yes/no) |
| `guestroom` | Object | Does the house include a guest room? (yes/no) |
| `basement` | Object | Does the house have a basement? (yes/no) |
| `hotwaterheating` | Object | Is hot water heating available? (yes/no) |
| `airconditioning` | Object | Is central air conditioning installed? (yes/no) |
| `parking` | Integer | Number of dedicated parking spots. |
| `prefarea` | Object | Is the house located in a preferred urban neighborhood? (yes/no) |
| `furnishingstatus` | Object | Furnishing condition of the house (furnished / semi-furnished / unfurnished). |

---

## Data Cleaning

Data cleaning is the initial defense against corrupted machine learning experiments. The cleaning pipeline consisted of:

1. **Dataset Ingestion & Inspection:** 
   The raw dataset was loaded into a Pandas DataFrame. Initial structural integrity checks were executed using:
   * `head()`: To visually verify data loading and check alignment.
   * `info()`: To evaluate memory footprint, row count, and data types.
   * `describe()`: To compute basic statistics (mean, standard deviation, quartiles) for numerical features.
2. **Missing Value Audit:** 
   We performed a systematic check using `.isnull().sum()` and `.isna().sum()`. The dataset contained **zero missing values**, indicating a complete dataset requiring no imputation.
3. **Duplicate Verification:** 
   Duplicate rows can artificially bias model training. Using `.duplicated().sum()`, we verified that the dataset contains no redundant records, ensuring that every observation represents a unique property transaction.
4. **Data Type Verification:** 
   Column data types were cross-referenced with their semantic descriptions. Numerical columns were confirmed as integers, and categorical variables were flagged for downstream encoding.

---

## Data Preprocessing

To make the dataset compatible with a linear equation ($Y = \beta_0 + \beta_1 X_1 + \dots + \beta_n X_n$), qualitative variables were transformed into quantitative inputs.

### 1. Binary Feature Encoding
Categorical variables with two levels (`yes` / `no`) were mapped to numeric boolean equivalents (`1` / `0`). This preserves the information while presenting it as a mathematical indicator variable.
* **Mapping Applied:** `yes` $\rightarrow$ `1`, `no` $\rightarrow$ `0`
* **Target columns:** `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, and `prefarea`.

### 2. Furnishing Status Encoding (One-Hot Encoding)
The column `furnishingstatus` contains three distinct nominal categories: `furnished`, `semi-furnished`, and `unfurnished`.
* We generated dummy variables for these categories using one-hot encoding.
* **Dummy Variable Trap Mitigation:** If we include three dummy columns, they will sum to exactly 1 for every row, introducing perfect multicollinearity. To prevent this mathematical redundancy, we dropped the `unfurnished` category. 
* **Final Mapping:** The baseline state is represented when `furnished = 0` and `semi-furnished = 0` (which implies the property is unfurnished).

---

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was carried out using Matplotlib and Seaborn to identify distributions, relationships, and structural properties within the dataset.

### 1. Outlier Analysis
Outliers were identified using visual boxplots and mathematically mapped using the Interquartile Range (IQR) method:

$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

* **Target Variables Evaluated:** `price`, `area`, `bedrooms`, `bathrooms`, `stories`, and `parking`.
* **Findings:** Outliers were detected in both `price` and `area`, representing high-value luxury estates.
* **Handling Strategy:** Rather than automatically removing these records, we manually inspected them. Because these records represent legitimate, high-value real estate configurations rather than data-entry errors, they were retained. Retaining them allows the model to learn the true premium tier of the housing market.

### 2. Feature Distribution Analysis
We generated histograms and Kernel Density Estimate (KDE) plots to examine the shapes of continuous variables:
* **Area Distribution:** The `area` feature shows a positive (right) skew. The majority of homes cluster between 2,500 and 6,000 square feet, with a long tail of large properties extending up to 16,000 square feet.
* **Discrete Features:** Features like `bedrooms`, `bathrooms`, `stories`, and `parking` show distinct distributions, reflecting typical residential property layouts (mostly 2–4 bedrooms and 1–2 bathrooms).

### 3. Target Variable Analysis
Analyzing the distribution of the target variable `price` revealed:
* A right-skewed distribution, which is common in real estate pricing.
* A peak near $3.5\text{M} - 4.5\text{M}$ units, stretching to a premium tail of houses priced up to $13.3\text{M}$ units.
* This skewness indicates that a future log-transformation of the target variable could be useful if model residuals show non-constant variance (heteroscedasticity).

### 4. Correlation Analysis
A Pearson correlation matrix and a matching heatmap were generated to examine linear relationships:
* **Strong Predictors:** `area` ($r \approx 0.53$), `bathrooms` ($r \approx 0.52$), and `airconditioning` ($r \approx 0.45$) show the strongest positive linear correlations with `price`.
* **Weak Predictors:** `hotwaterheating` ($r \approx 0.09$) and `semi-furnished` ($r \approx 0.06$) display weak linear relationships with the target variable.

---

## Multicollinearity Analysis (VIF)

Linear regression requires that predictors are not highly collinear. To assess this, we calculated the **Variance Inflation Factor (VIF)** for each numerical predictor:

$$\text{VIF}_i = \frac{1}{1 - R_i^2}$$

Where $R_i^2$ is the coefficient of determination obtained by regressing predictor $X_i$ against all other predictors.

* **Analysis Scope:** `area`, `bedrooms`, `bathrooms`, `stories`, `parking`, and the encoded categorical variables.
* **Findings:**
  * Dropping `unfurnished` resolved the perfect multicollinearity introduced by the dummy variable trap.
  * The remaining predictors showed VIF values well below the standard threshold of 5.0, confirming they are suitable for linear regression modeling.
  * The small amount of natural multicollinearity present (e.g., properties with more bedrooms often have more bathrooms and larger areas) is normal and was retained to maintain model interpretability.

---

## Feature Selection

Based on our EDA, correlation matrix, and VIF analysis, we finalized the features for the regression model:
1. **Retained Features:** `area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, `furnished`, and `semi-furnished`.
2. **Dropped Features:** `unfurnished` (to prevent perfect collinearity).

---

## Clean Dataset Export

Once the pipeline completed, the preprocessed and encoded DataFrame was exported:
* **Output Path:** `data/housing_cleaned.csv`
* **Format:** Comma-Separated Values (CSV), written via Pandas `.to_csv(index=False)`.
* **Status:** Verified and ready for Phase 2 modeling.

---

## Key Findings

* **Data Quality:** The Kaggle dataset is well-maintained, with no missing values or duplicate entries.
* **Key Pricing Drivers:** Property size (`area`) and structural features (like the number of `bathrooms` and `airconditioning`) show the strongest positive relationships with house prices.
* **Outliers:** Outliers in `price` and `area` represent valid luxury homes and were kept to help the model learn the high-end market segment.
* **Collinearity Status:** Resolving the dummy variable trap reduced VIF values below 5.0, confirming the feature set is ready for stable regression modeling.

## Technologies Used

* **Python 3.10+**: Core programming language.
* **Pandas**: Structured data manipulation and processing.
* **NumPy**: Linear algebra and multi-dimensional array calculations.
* **Matplotlib & Seaborn**: Data visualization and statistical graphics.
* **Statsmodels**: Variance Inflation Factor (VIF) diagnostics.

---

## Author

* Email: preethamdeoxys@gmail.com (mailto:preethamdeoxys@gmail.com)
* GitHub: @zgander (https://github.com/zgander)
* LinkedIn: Preetham Vasireddy (https://www.linkedin.com/in/preetham-vasireddy-967924327/)
