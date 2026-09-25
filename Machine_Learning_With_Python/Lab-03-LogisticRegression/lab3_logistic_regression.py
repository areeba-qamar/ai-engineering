# Import Pandas for creating and working with tabular data
import pandas as pd

# Import train_test_split to divide the data into training and testing sets
from sklearn.model_selection import train_test_split

# Import LogisticRegression to create our classification model
from sklearn.linear_model import LogisticRegression

# Import accuracy_score to measure how many predictions were correct
from sklearn.metrics import accuracy_score


# Create a small dataset containing student information
# study_hours and attendance are the input features
# admission is the target: 1 = Admitted, 0 = Not Admitted
data = {
    "study_hours": [1, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 7, 8, 9],
    "attendance": [55, 60, 65, 68, 70, 72, 78, 80, 82, 88, 92, 95],
    "admission": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

# Convert the dictionary into a Pandas DataFrame
df = pd.DataFrame(data)

# Display the complete dataset
print(df)


# Separate the independent variables (X) from the dependent variable (y)
# X contains the features that the model will use to make predictions
X = df[["study_hours", "attendance"]]

# y contains the target/class that we want the model to predict
# 0 = Not Admitted
# 1 = Admitted
y = df["admission"]

print("\nX:")
print(X)

print("\ny:")
print(y)


# Split the data into training and testing sets
# 80% of the data will be used to train the model
# 20% will be used to test the model on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data:")
print(X_train)

print("\nTesting data:")
print(X_test)


# Create the Logistic Regression model
# The model will learn the relationship between the input features
# and the probability of a student being admitted
model = LogisticRegression()

print("\nModel created successfully!")


# Train the model using the training data
# During training, the model learns patterns from study hours and attendance
# and learns how they relate to the admission outcome
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Predict the class (0 or 1) for the unseen test data
# The model uses the learned patterns to classify each test student
y_pred = model.predict(X_test)

print("\nPredicted classes:")
print(y_pred)


# Predict the probability of each class for the test data
# predict_proba() gives probabilities for:
# column 0 = probability of class 0 (Not Admitted)
# column 1 = probability of class 1 (Admitted)
probabilities = model.predict_proba(X_test)

print("\nPredicted probabilities:")
print(probabilities)


# Evaluate the model using accuracy
# Accuracy tells us what proportion of predictions were correct
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# Compare the actual classes with the predicted classes
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results)