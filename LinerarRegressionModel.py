# Import all necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Prepare the dataset
data = {
    'position': [1, 2, 3, 4, 5],
    'GDP': [100, 80, 60, 40, 30]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the data
X = df[['position']]
y = df[['GDP']]

# Create a model
model = LinearRegression()

# Finding the relationship between X and y
model.fit(X, y)

# Getting the model parameters -> slope and intercept
slope = model.coef_[0][0]  # Extracting the single value from the array
intercept = model.intercept_[0]
print("The slope is:", slope)
print("The intercept is:", intercept)

# Predicting the new GDP for the next position (e.g., position 6)
position_to_predict = pd.DataFrame({'position': [6]})
predicted_gdp = model.predict(position_to_predict)
print(f"Predicted GDP for position {position_to_predict.iloc[0, 0]}: {predicted_gdp[0][0]}")

# Create a range of positions for plotting the regression line
X_range = np.linspace(1, 6, 100).reshape(-1, 1)  # Extend range to include new position
y_range = model.predict(X_range)

# Plot the data points
plt.scatter(df['position'], df['GDP'], color='blue', label='Data points')

# Plot the regression line
plt.plot(X_range, y_range, color='red', label='Regression line')

# Add labels and title
plt.xlabel('Position')
plt.ylabel('GDP')
plt.title('GDP vs. Position')
plt.legend()

# Show the plot
plt.show()
