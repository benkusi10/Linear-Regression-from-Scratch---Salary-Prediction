# Linear-Regression-from-Scratch---Salary-Prediction
Learn how to implement a Linear Regression Model in python using just Maths and Numpy. 
Linear regression is a fundamental machine learning algorithm used for predicting a continuous target variable based on one or more input features. This implementation demonstrates the inner workings of linear regression by building it from the ground up, making it an excellent educational resource for understanding:

The mathematics behind linear regression

How gradient descent optimizes model parameters

The importance of data preprocessing and visualization

Model evaluation through cost function analysis

The model uses the Salary Dataset which contains:

YearsExperience: Number of years of professional experience

Salary: Corresponding annual salary in dollars

The dataset consists of 30 samples with a clear positive correlation between experience and salary, making it ideal for demonstrating linear regression.

🔧 Implementation Details
Key Components
1. Cost Function (cost_function)
Implements Mean Squared Error (MSE) with a 1/2 factor for mathematical convenience

Formula: J(w,b) = (1/2m) * Σ(f(xⁱ) - yⁱ)²

Measures how well the model fits the training data

2. Gradient Function (gradient_function)
Computes partial derivatives of the cost function with respect to parameters

∂J/∂w = (1/m) * Σ(f(xⁱ) - yⁱ) * xⁱ

∂J/∂b = (1/m) * Σ(f(xⁱ) - yⁱ)

3. Gradient Descent (gradient_descent)
Iteratively updates parameters to minimize the cost function

Update rules:

w = w - α * ∂J/∂w

b = b - α * ∂J/∂b

Tracks cost reduction across iterations

Parameters
Learning Rate (α): 0.01 - Controls the step size during gradient descent

Iterations: 10,000 - Number of training epochs

Initial Parameters: w = 0, b = 0

📈 Results
After training for 10,000 iterations, the model achieved:

Final Weight (w): 9449.96

Final Bias (b): 25792.20

The cost function shows consistent decrease throughout training:

Initial Cost: ~1.34 billion

Final Cost: ~15.6 million

Visualization
The notebook includes two key visualizations:

Scatter Plot: Raw data points showing the relationship between experience and salary

Regression Line: The learned linear relationship overlaid on the data points
