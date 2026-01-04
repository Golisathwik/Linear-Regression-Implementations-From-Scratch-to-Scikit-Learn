# Linear-Regression-Implementations-From-Scratch-to-Scikit-Learn
This repository contains three distinct approaches to building a Linear Regression model. Linear Regression is a fundamental machine learning algorithm used to quantify the relationship between a dependent variable (Y) and an independent variable (X), allowing us to make predictions based on that relationship.
The core objective is to find the "best-fitting" line, represented by the linear equation:

Where:

*  is the **Slope** (the weight or coefficient).
*  is the **Y-Intercept** (the bias).

---

# Linear Regression Implementations: From Scratch to Scikit-Learn

This repository demonstrates three distinct approaches to implementing Linear Regression to predict **House Prices** based on **Size (sq ft)**. The goal is to move from fundamental mathematical derivation to algorithmic optimization, and finally to professional library implementation.

## 📂 Dataset Overview
The model is trained on a custom dataset (`predict.csv`) containing the following relationship:
* **Independent Variable ($X$):** House Size (in sq ft)
* **Dependent Variable ($Y$):** Price (in Rupees)

## 🚀 Implementation 1: Simple Linear Regression (Least Squares)
**Method:** Closed-form Mathematical Solution (Ordinary Least Squares)

This approach calculates the exact "line of best fit" without any iteration or loops. It uses statistics to find the global minimum of error instantly. This is ideal for small datasets where we can mathematically derive the perfect slope ($m$) and intercept ($b$).

### 📝 Step-by-Step Derivation of $m$ and $b$
To find the equation $Y = mX + b$, we follow this 4-step mathematical process:

**Step 1: Calculate the Means**
First, we find the average house size ($\bar{X}$) and average price ($\bar{Y}$).

$$\bar{X} = \frac{\sum_{i=1}^{n} X_i}{n} \quad \text{and} \quad \bar{Y} = \frac{\sum_{i=1}^{n} Y_i}{n}$$

**Step 2: Calculate Deviations**
We measure how far each house's size and price are from the average.
* $X_{diff} = (X_i - \bar{X})$
* $Y_{diff} = (Y_i - \bar{Y})$

**Step 3: Calculate the Slope ($m$)**
The slope represents the price increase per square foot. It is calculated by dividing the sum of the product of deviations by the squared deviations of $X$.

$$m = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2}$$

**Step 4: Calculate the Intercept ($b$)**
Now that we have the slope ($m$), we can solve for $b$ (the base price) using the mean values.

$$b = \bar{Y} - (m \times \bar{X})$$


### Key Insight
This method guarantees the mathematically lowest possible error for the training data but is computationally expensive for massive datasets with millions of rows.

---

## 🚀 Implementation 2: Linear Regression from Scratch (Gradient Descent)
**Method:** Iterative Optimization (Steepest Descent)


Unlike the first method, which solves the problem instantly, this approach "learns" the best line over time. It starts with a random guess and iteratively refines the slope ($m$) and intercept ($b$) to minimize the error.

### ⚠️ A Note on Hyperparameters
* **Learning Rate (`L`):** `0.000001`
* **Reasoning:** In this dataset, the target values (Price) are large (up to 170,000). When calculating the Mean Squared Error (MSE), these large numbers are squared, resulting in massive values.
    * If the Learning Rate is too high (e.g., 0.01), the algorithm tries to take huge steps, causing the error to "explode" (overshoot) and the numbers to become unstable or infinite.
    * We use a very small learning rate (`1e-6`) to ensure the model takes tiny, stable steps towards the minimum error without crashing.

### ⚙️ The Training Loop (Step-by-Step)
The model optimizes $m$ and $b$ over **2000 epochs** using the following logic:

**Step 1: Initialization**
We start with $m=0$ and $b=0$.

**Step 2: Calculate Predictions ($Y_{pred}$)**
For every data point, we calculate what the current line predicts:
$$Y_{pred} = mX + b$$

**Step 3: Calculate the Gradient (The Direction)**
We need to know which way to move $m$ and $b$ to reduce the error. We find the partial derivative of the MSE function:

* **Gradient for $m$:**
    $$D_m = -\frac{2}{n} \sum X (Y_{actual} - Y_{pred})$$
* **Gradient for $b$:**
    $$D_b = -\frac{2}{n} \sum (Y_{actual} - Y_{pred})$$

**Step 4: Update Weights (The Step)**
We adjust the current values by moving in the opposite direction of the gradient, scaled by the Learning Rate ($L$):
$$m_{new} = m_{current} - (L \times D_m)$$
$$b_{new} = b_{current} - (L \times D_b)$$

**Step 5: Repeat**
This process repeats 2000 times until the line settles on the optimal fit.
