# Linear-Regression-Implementations-From-Scratch-to-Scikit-Learn
This repository contains three distinct approaches to building a Linear Regression model. Linear Regression is a fundamental machine learning algorithm used to quantify the relationship between a dependent variable (Y) and an independent variable (X), allowing us to make predictions based on that relationship.
The core objective is to find the "best-fitting" line, represented by the linear equation:

Where:

*  is the **Slope** (the weight or coefficient).
*  is the **Y-Intercept** (the bias).

---

# Linear Regression: From Mathematical Formulas to Industry Standards

This repository demonstrates three distinct approaches to implementing Linear Regression to predict **House Prices** based on **Size (sq ft)**. The goal is to move from fundamental mathematical derivation to algorithmic optimization, and finally to professional library implementation.

## 📂 Dataset Overview

The model is trained on a custom dataset (`predict.csv`) containing the following relationship:

* **Independent Variable ():** House Size (in sq ft)
* **Dependent Variable ():** Price (in Rupees)

## 🚀 Implementation 1: Simple Linear Regression (Least Squares)

**Method:** Closed-form Mathematical Solution (Ordinary Least Squares)

This approach calculates the exact "line of best fit" without any iteration or loops. It uses statistics to find the global minimum of error instantly. This is ideal for small datasets where we can mathematically derive the perfect slope () and intercept ().

### 📝 Step-by-Step Derivation of  and 

To find the equation , we follow this 4-step mathematical process:

**Step 1: Calculate the Means**
First, we find the average house size () and average price ().


**Step 2: Calculate Deviations**
We measure how far each house's size and price are from the average.

* 
* 

**Step 3: Calculate the Slope ()**
The slope represents the price increase per square foot. It is calculated by dividing the sum of the product of deviations by the squared deviations of .

**Step 4: Calculate the Intercept ()**
Now that we have the slope (), we can solve for  (the base price) using the mean values.

---

### Key Insight

This method guarantees the mathematically lowest possible error for the training data but is computationally expensive for massive datasets with millions of rows.

---

### Next Step

Shall I provide the content for **Implementation 2: Gradient Descent (From Scratch)** next?

(This is where I will include your specific note about the **Learning Rate (0.000001)** and the reasoning regarding the large MSE values.)
