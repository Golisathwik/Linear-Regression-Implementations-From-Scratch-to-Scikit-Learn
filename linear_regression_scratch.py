import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('D:/sathwik files/ml practice/linear regression/predict.csv')
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i]['size']
        y = points.iloc[i]['price']
        total_error += (y - (m * x + b))**2
    return total_error / float (len(points))
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i]['size']
        y = points.iloc[i]['price']
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now)) 
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L 
    return m, b

def predict_function(m, b, x ):
    return m * x + b
m = 0
b = 0
L = 0.000001
epochs= 2000
x= int(input('Enter the size of the area: '))
for i in range(epochs):
    current_loss = loss_function(m, b, data)
    if i % 50 == 0:
        print(f"Epoch: {i} | Error: {current_loss}")
    m, b = gradient_descent(m, b, data, L)
print(m, b)
result= predict_function(m,b,x)
print('the house price is: Rs.', int(result))
plt.scatter(data['size'], data['price'], color="black", marker= '+')
plt.plot(list(range (100, 1001)), [m * x + b for x in range(100, 1001)], color="red")
plt.show()