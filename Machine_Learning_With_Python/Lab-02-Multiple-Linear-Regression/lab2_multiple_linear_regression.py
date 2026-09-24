# Import Pandas for creating and working with tabular data
import pandas as pd

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# Import LinearRegression to create a multiple linear regression model
from sklearn.linear_model import LinearRegression


# Create a larger dataset containing house information and prices
data = {
    "size": [
        750, 900, 1000, 1100, 1250,
        1400, 1500, 1650, 1800, 1950,
        2100, 2250, 2400, 2600, 2800
    ],
    "bedrooms": [
        2, 2, 2, 3, 3,
        3, 3, 3, 4, 4,
        4, 4, 5, 5, 5
    ],
    "age": [
        15, 12, 10, 9, 8,
        7, 6, 5, 5, 4,
        3, 3, 2, 2, 1
    ],
    "price": [
        115, 130, 145, 160, 175,
        195, 205, 225, 245, 265,
        285, 305, 330, 355, 385
    ]
}

# Convert the data into a Pandas DataFrame
df = pd.DataFrame(data)

print(df)


# Separate the independent variables (X) and dependent variable (y)
# X contains multiple inputs used to predict the house price
X = df[["size", "bedrooms", "age"]]

# y is the target/output that the model will predict
y = df["price"]

print("\nX:")
print(X)

print("\ny:")
print(y)


# Split the data into training and testing sets
# 80% of the data is used for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:")
print(X_train)

print("\nTesting data:")
print(X_test)


# Create a multiple linear regression model
model = LinearRegression()

print("\nModel created successfully!")


# Train the model using the training data
# The model learns how size, bedrooms, and age relate to house price
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Use the trained model to predict prices for the test data
y_pred = model.predict(X_test)

print("\nPredicted prices:")
print(y_pred)


# Display the coefficients learned by the model
# Each coefficient represents the effect of one input variable
print("\nCoefficients:")
print("Size:", model.coef_[0])
print("Bedrooms:", model.coef_[1])
print("Age:", model.coef_[2])

# Display the intercept of the model
print("\nIntercept:", model.intercept_)


# Evaluate the model using the test data
# R² indicates how well the model explains the variation in the target values
r2_score = model.score(X_test, y_test)

print("\nR² Score:", r2_score)


# Compare the actual prices with the model's predicted prices
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted:")
print(results)