import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Data
data = pd.read_csv('D:/sathwik files/ml practice/linear regression/predict.csv')

# Extract lists for easier calculation
X = data['size'].values
Y = data['price'].values
n = len(X)

# --- Mathematical Implementation (Least Squares) ---

# Step 1: Calculate the Means
x_mean = sum(X) / n
y_mean = sum(Y) / n

# Step 2: Calculate the terms needed for the numerator and denominator
numerator = 0
denominator = 0

for i in range(n):
    # Sum of (x - x_mean) * (y - y_mean)
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    
    # Sum of (x - x_mean)^2
    denominator += (X[i] - x_mean) ** 2

# Step 3: Calculate Slope (m) and Intercept (b)
m = numerator / denominator
b = y_mean - (m * x_mean)

print(f"Calculated Slope (m): {m}")
print(f"Calculated Intercept (b): {b}")

# --- Prediction & Visualization ---

# Prediction Function
def predict(x_input):
    return m * x_input + b

# Take user input
user_size = int(input("Enter the size of the area: "))
predicted_price = predict(user_size)
print(f"The estimated house price is: Rs. {int(predicted_price)}")

# Plotting the Regression Line
plt.scatter(X, Y, color="black", marker='+', label="Data Points")
plt.plot(X, [predict(x) for x in X], color="blue", label="Regression Line")
plt.xlabel('Size (sq ft)')
plt.ylabel('Price')
plt.title('Simple Linear Regression (Least Squares Method)')
plt.legend()
plt.show()