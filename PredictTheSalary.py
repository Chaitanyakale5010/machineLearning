# import libreries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# prepare a dataset 
data = {
        'exp':[30,25,20,10,5,2],
        'salary':[100,77,45,30,10,5]
       }

# create a table 
df = pd.DataFrame(data)

# Split the datset into x(Independent) and y(dependent) variables 
x = df[['exp']]
y = df[['salary']]

# create a model
model = LinearRegression()

# draw out the relation bwtween the variables x and y
model.fit(x,y)

#Find the values 
slope = model.coef_[0]
intercept = model.intercept_

print("The slope is:",slope)
print("The intercept is:",intercept)

new_exp = pd.DataFrame({'exp':[20]})
New_Predicted_Salery = model.predict(new_exp)
print(New_Predicted_Salery)

X_range = np.linspace(1, 6, 100).reshape(-1, 1)  # Extend range to include new position
y_range = model.predict(X_range)

# Plot the data points
plt.scatter(df['exp'], df['salary'], color='blue', label='Data points')

# Plot the regression line
plt.plot(X_range, y_range, color='red', label='Regression line')

# Add labels and title
plt.xlabel('salary')
plt.ylabel('exp')
plt.title('exp vs. salery')
plt.legend()

# Show the plot
plt.show()
