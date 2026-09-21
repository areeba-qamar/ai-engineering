# Import Pandas for creating and working with tabular data
import pandas as pd

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# Import LinearRegression to create a linear regression model
from sklearn.linear_model import LinearRegression


# Create a small dataset containing house size and price
data = {
    "size": [800, 1000, 1200, 1500, 1800, 2000],
    "price": [120, 150, 180, 220, 270, 300]
}

# Convert the data into a Pandas DataFrame
df = pd.DataFrame(data)

print(df)


# Separate the independent variable (X) and dependent variable (y)
# X is the input used to predict the house price
X = df[["size"]]

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


# Create a simple linear regression model
model = LinearRegression()

print("\nModel created successfully!")


# Train the model using the training data
# The model learns the relationship between house size and price
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Use the trained model to predict prices for the test data
y_pred = model.predict(X_test)

print("\nPredicted prices:")
print(y_pred)


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

