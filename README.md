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

---

### Key Insight
This method guarantees the mathematically lowest possible error for the training data but is computationally expensive for massive datasets with millions of rows.
