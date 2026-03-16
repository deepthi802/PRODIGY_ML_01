import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("house_data.csv")

# Display first few rows
print("Dataset Preview:")
print(data.head())

# Features (Independent Variables)
X = data[['SquareFeet', 'Bedrooms', 'Bathrooms']]

# Target (Dependent Variable)
y = data['Price']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)
# Predict house prices
y_pred = model.predict(X_test)
# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
example = np.array([[2000, 3, 2]])  # 2000 sqft, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(example)

print("\nExample Prediction:")
print("Predicted price for 2000 sqft house:", predicted_price[0])
