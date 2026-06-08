# Used Car Price Prediction

## About the Project

This project predicts the price of a used car using Multiple Linear Regression implemented from scratch with NumPy.

The main goal of this project was to understand how linear regression works internally instead of using Scikit-Learn's built-in LinearRegression model.

The model is trained using Gradient Descent and evaluated using MAE, RMSE and R² Score.

---

## Dataset

The dataset contains information about used cars such as:

- Brand
- Model Year
- Mileage
- Fuel Type
- Transmission Type
- Price

The target variable is the car price.

---

## Data Preprocessing

Before training the model, the following preprocessing steps were performed:

- Removed unnecessary columns
- Filled missing values in fuel_type using mode
- Converted mileage and price columns into numeric values
- Applied one-hot encoding to categorical features
- Converted model year into car age
- Normalized the features using mean and standard deviation

---

## Model

The Multiple Linear Regression algorithm was implemented manually using NumPy.

The implementation includes:

- Weight initialization
- Prediction using linear equation
- Mean Squared Error loss function
- Gradient Descent optimization
- Weight and bias updates

No machine learning libraries were used for training the model.

---

## Evaluation Metrics

The model was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Visualization

A scatter plot is generated to compare actual prices with predicted prices.

The closer the points are to the diagonal line, the better the predictions.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

## Files

```text
improvedUCP.py
used_cars.csv
README.md
```

---

## What I Learned

Through this project I learned:

- Data preprocessing
- Feature engineering
- One-hot encoding
- Feature normalization
- Gradient Descent
- Multiple Linear Regression
- Model evaluation techniques

---

## Future Improvements

- Random train-test split
- Better feature engineering
- Compare results with Scikit-Learn
- Try Ridge and Lasso Regression
- Deploy as a web application

---

## Author

Joel Suman

Computer Science Engineering Student
