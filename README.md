# Boston Housing Price Prediction with Random Forest Regressor
## Overview
This project focuses on predicting housing prices in Boston using a Random Forest Regressor. The dataset used contains various features related to housing and serves as the basis for making accurate predictions.

## Learning Outcomes
1. Implementing a Random Forest Regressor for regression tasks.
2. Understanding and evaluating regression metrics like R^2 score, Mean Absolute Error (MAE), and Mean Squared Error (MSE).
3. Analyzing and interpreting the difference between actual and predicted values.
4. Gaining insights into the impact of feature selection on regression models.

## Installation
Clone the repository: git clone https://github.com/your_username/boston-housing-prediction.git
Install the required packages: pip install pandas numpy scikit-learn

## Usage
Ensure you have Python and pip installed on your system.
Install the necessary packages
Run the main Python script main.py:
The script will perform the following tasks:
  Load and preprocess the dataset.
  Train a Random Forest Regressor on the data.
  Evaluate the model's performance using R^2 score, Mean Absolute Error (MAE), and Mean Squared Error (MSE).
  Display a comparison between actual and predicted housing prices.

## File Descriptions
boston.csv: Dataset containing housing-related features and target prices.
main.py: Python script for training the Random Forest Regressor and evaluating the model.

## Output Example

r2 score: 0.865
Mean absolute error: 2.345
   Actual value  Predicted value  difference
0          24.0            22.84        1.16
1          25.0            29.12       -4.12
2          15.6            17.88       -2.28
3          18.3            20.46       -2.16
...

Mean squared error: 10.12
R^2 accuracy: 0.76
MAE accuracy: -3.22
MSE accuracy: -21.54

Acknowledgments:
This program was created as part of the AI, DS and ML course offered by the National Emerging Skills Program (NESP).


