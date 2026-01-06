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

---

## 🚀 Implementation 3: Linear Regression using Scikit-Learn
**Method:** Industry Standard Library

This version utilizes `sklearn.linear_model.LinearRegression`. In a professional setting, we rarely calculate gradients manually. Instead, we use optimized libraries like Scikit-Learn, which handle the underlying mathematics using highly efficient C-based backends (LAPACK/BLAS).

### 💻 Implementation Workflow
The code follows the standard "Import -> Instantiate -> Fit -> Predict" lifecycle:

**Step 1: Data Preparation (Reshaping)**
Scikit-Learn models expect the input features ($X$) to be a **2D array** (a matrix), even if there is only one feature.
* *Code:* `x_size = file[['size']]`
* *Note:* The double brackets `[[ ]]` ensure pandas returns a DataFrame (2D) rather than a Series (1D).

**Step 2: Model Instantiation**
We create an instance of the Linear Regression class.
* *Code:* `model = LinearRegression()`

**Step 3: Training (The "Fit" Method)**
This is where the magic happens. The `.fit()` method calculates the optimal $m$ and $b$ internally.
* *Code:* `model.fit(x_size, y_price)`

**Step 4: Extraction & Prediction**
Once trained, we can extract the learned parameters or make predictions.
* **Slope ($m$):** Accessed via `model.coef_`
* **Intercept ($b$):** Accessed via `model.intercept_`
* **Prediction:** `model.predict(new_data)`

---

## 📊 Comparison of Approaches

| Feature | Simple Linear Regression | Gradient Descent (From Scratch) | Scikit-Learn |
| :--- | :--- | :--- | :--- |
| **Complexity** | Low (Basic Math) | High (Calculus & Loops) | Low (Abstracted) |
| **Accuracy** | Exact | Approximate (depends on epochs) | Exact |
| **Speed** | Fast for small data | Slow (Python loops) | Very Fast (Optimized C) |
| **Use Case** | Understanding Stats | Understanding Neural Networks | **Real-world Projects** |

---

## 🧮 Multiple Linear Regression: The Matrix Approach

In Simple Linear Regression ($y = mx + c$), we use basic algebra to find the slope and intercept. However, when we have **multiple variables** ($x_1, x_2, x_3 \dots$), standard algebra becomes messy and inefficient.

Instead, we use **Linear Algebra (Matrices)** to solve for all coefficients simultaneously.

### 1. The Mathematical Model
Instead of a simple line, we are now defining a **hyperplane**:

$$y = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n$$

Where:
* **$y$:** Predicted value (e.g., House Price).
* **$b_0$:** Intercept (Bias).
* **$b_1, \dots, b_n$:** Coefficients (Weights) for each feature.
* **$x_1, \dots, x_n$:** Features (e.g., Size, Bedrooms, Age).

---

### 2. The Matrix Setup
To calculate this efficiently, we arrange our data into three key matrices.

#### A. The Feature Matrix ($X$)
We add a **column of 1s** to the start of our features. This acts as the placeholder for the intercept ($b_0$).

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?X=\begin{bmatrix}1&x_{11}&x_{12}\\1&x_{21}&x_{22}\\1&x_{31}&x_{32}\\\vdots&\vdots&\vdots\end{bmatrix}" alt="Feature Matrix X" />
</p>

*(Rows represent individual data samples, Columns represent features like Intercept, Size, Bedrooms)*

#### B. The Target Vector ($Y$)
This contains the actual values we are trying to predict.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?Y=\begin{bmatrix}y_1\\y_2\\y_3\\\vdots\end{bmatrix}" alt="Target Vector Y" />
</p>

#### C. The Coefficients Vector ($B$)
This is what the model is trying to calculate ($b_0, b_1, b_2$).

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?B=\begin{bmatrix}b_0\\b_1\\b_2\\\vdots\end{bmatrix}" alt="Coefficients Vector B" />
</p>

---

## 🛠️ How to Run
1.  Clone the repository.
2.  Ensure `predict.csv` is in the correct directory.
3.  Run the desired script:
    ```bash
    python linear_regression_scratch.py
    # or
    python linear_regression_sklearn.py
    # or
    python simple_linear_regression.py
    ```
