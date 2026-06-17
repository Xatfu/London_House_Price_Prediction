# London Housing Price Prediction

## Project Overview
This project aims to predict London housing prices using a dataset containing various property attributes. The goal is to build a robust machine learning model that can accurately estimate property values based on features like location, number of rooms, floor area, and energy ratings.

## Data Overview
The dataset consists of a training set (`train.csv`) and a test set (`test.csv`), along with a sample submission file (`sample_submission.csv`). Key features include:
- `ID`: Unique identifier for each property.
- `fullAddress`: Full address of the property.
- `postcode`: Postcode of the property.
- `country`: Country (all London, so mostly 'England').
- `outcode`: Outer postcode district.
- `latitude`, `longitude`: Geographical coordinates.
- `bathrooms`, `bedrooms`, `livingRooms`: Number of bathrooms, bedrooms, and living rooms.
- `floorAreaSqM`: Floor area in square meters.
- `tenure`: Type of property tenure (Leasehold, Freehold, etc.).
- `propertyType`: Type of property (Flat, House, Bungalow, etc.).
- `currentEnergyRating`: Energy efficiency rating (A-G).
- `sale_month`, `sale_year`: Month and year of sale.
- `price`: Sale price (target variable in `df_train`).

## Methodology

### 1. Data Loading and Initial Inspection
- You can find the dataset on Kaggle here - https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques
- Loaded `train.csv`, `test.csv`, and `sample_submission.csv` using pandas.
- Performed initial inspection using `df.info()` and `df.isna().sum()` to understand data types and identify missing values.

### 2. Feature Dropping
- Removed `ID`, `fullAddress`, `country`, and `postcode` columns as they were either redundant or directly handled by other features (`outcode`) and not directly useful for modeling.

### 3. Handling Missing Values
- **`bathrooms` and `bedrooms`**: Imputed missing values using random sampling based on the observed distribution of these features in the non-missing data. This introduces some variability but maintains the distribution.
- **`livingRooms` and `floorAreaSqM`**:
    - Created binary indicator columns (`rooms_is_missing`, `area_is_missing`) to capture the fact of missingness.
    - Filled missing `livingRooms` with 1.0 (a common default).
    - Filled missing `floorAreaSqM` by grouping by `livingRooms` and using the median `floorAreaSqM` for each group. Remaining `floorAreaSqM` NaNs were filled with the overall median.
- **`tenure`**: Imputed missing values using random sampling between 'Leasehold' and 'Freehold' based on their approximate proportions.
- **`propertyType`**: Filled missing values with 'Purpose Built Flat', which is the most frequent category.
- **`currentEnergyRating`**:
    - Created an indicator column (`energy_is_missing`).
    - Filled missing values with 'D' (a common energy rating).
    - Mapped energy ratings (A-G) to numerical values (7-1) for ordinal encoding.

### 4. Feature Engineering
- **One-Hot Encoding**: Applied one-hot encoding to categorical features (`outcode`, `tenure`, `propertyType`) using `pd.get_dummies`. `drop_first=True` was used to avoid multicollinearity.
- **Alignment**: Aligned the `df_train` and `df_test` dataframes after one-hot encoding to ensure they have the same columns, filling missing columns with 0.

### 5. Model Training
- **Pipeline**: Used a scikit-learn `Pipeline` consisting of:
    - `RobustScaler`: To scale features, making the model less sensitive to outliers.
    - `SGDRegressor`: A Stochastic Gradient Descent Regressor, suitable for large datasets.
- **Hyperparameter Tuning**: Employed `GridSearchCV` to find the best hyperparameters for the `SGDRegressor`.
    - `param_grid` included `penalty` ('l2', 'l1'), `alpha` (0.0001, 0.1), and `learning_rate` ('invscaling', 'adaptive').
    - Cross-validation was set to `cv=3`.
- **Training**: The `GridSearchCV` was fitted on the `X_train` and `y_train` data.

### 6. Prediction and Submission
- Used the best model found by `GridSearchCV` to predict prices on the `X_test` dataset.
- The predicted prices were then assigned to the 'price' column of the `df_sub` (sample submission) dataframe.
- The `df_sub` dataframe was saved to a CSV file named `sub.csv` for submission.
