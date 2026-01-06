import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('multi_linear_predict.csv')
x= data[['size','Bedrooms','Floors']].values
y= data['Price_Lakhs'].values
model= LinearRegression()
model.fit(x,y)
print("Model Trained!")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
x_size= int(input("enter the house size: "))
x_room= int(input("enter the no of bedrooms: "))
x_floor= int(input("enter the no of floors: "))

x_input= [[x_size,x_room,x_floor]]
predict= model.predict(x_input)
print(int(predict[0]),' lakhs')
x_predict = model.predict(x)

plt.scatter(data['size'],y, color= 'red')
plt.scatter(data['Bedrooms'],y, color= 'green')
plt.scatter(data['Floors'],y, color= 'blue')
plt.plot(x, x_predict)
plt.title("All Variables vs Price (Note the Scale Difference!)")
plt.xlabel("Input Value")
plt.ylabel("Price")
plt.show()