# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Set a random seed for reproducibility
np.random.seed(seed=1)

# Load the Boston housing dataset
boston = pd.read_csv('boston.csv')

# Separate features (x) and target (y)
x = boston.drop('MEDV', axis=1)
y = boston['MEDV']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initialize and train a RandomForestRegressor
boston_model = RandomForestRegressor()
boston_model.fit(x_train, y_train)

# Predict the target values
y_predicted = boston_model.predict(x_test)

# Calculate and print R^2 score
from sklearn.metrics import r2_score
print(f'r2 score: {r2_score(y_test, y_predicted)}')

# Calculate and print Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_predicted)
print(f'Mean absolute error: {mae}')

# Create a DataFrame to compare actual and predicted values
df = pd.DataFrame(data={'Actual value': y_test,
                        'Predicted value': y_predicted})

# Calculate the difference between actual and predicted values
df['difference'] = df['Actual value'] - df['Predicted value']
print(df)

# Calculate and print Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_predicted)
print(f'Mean squared error: {mse}')

# Perform cross-validation and calculate R^2, MAE, and MSE accuracies
from sklearn.model_selection import cross_val_score
cross_acc = cross_val_score(boston_model, x, y, cv=5, scoring='r2')
print(f'R^2 accuracy: {cross_acc.mean()}')
cross_acc = cross_val_score(boston_model, x, y, cv=5, scoring='neg_mean_absolute_error')
print(f'MAE accuracy: {cross_acc.mean()}')
cross_acc = cross_val_score(boston_model, x, y, cv=5, scoring='neg_mean_squared_error')
print(f'MSE accuracy: {cross_acc.mean()}')
