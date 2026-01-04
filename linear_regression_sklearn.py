import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

# Input: House Size (sq ft)
# We use double brackets [[ ]] because models expect a list of lists (rows and columns)
file = pd.read_csv('predict.csv')
x_size = file[['size']]  
y_price = file['price']
# 2. Create the Model
model = LinearRegression()
size= int(input("enter the size? "))
# 3. Train the Model (The "Cooking" step)
model.fit(x_size, y_price)# Check what the model learned
print(f"Slope (m): {model.coef_}")
print(f"Intercept (c): {model.intercept_}")# Predict price for a 4000 sq ft house
predicted_price = model.predict(pd.DataFrame([[size]], columns=['size']))
predicted_size= model.predict(x_size)
print(int(predicted_price[0]),'Rs')
plt.scatter(x_size,y_price, color='red', marker= '+')
plt.plot(x_size, predicted_size, marker='o')
plt.title('house price prediction')
plt.xlabel('size')
plt.ylabel('price')
plt.show()