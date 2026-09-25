# Import Pandas for displaying and working with tabular data
import pandas as pd

# Import the Iris dataset from scikit-learn
from sklearn.datasets import load_iris

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# Import LogisticRegression as the base binary classifier
from sklearn.linear_model import LogisticRegression

# Import OneVsRestClassifier for multi-class One-vs-Rest classification
from sklearn.multiclass import OneVsRestClassifier

# Import accuracy_score to evaluate the model
from sklearn.metrics import accuracy_score

# Import confusion_matrix to see correct and incorrect predictions by class
from sklearn.metrics import confusion_matrix


# Load the Iris dataset
# The Iris dataset contains measurements of flowers from three different species
iris = load_iris()


# Create a Pandas DataFrame containing the four flower measurements
# These will be our input features (X)
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add the target species to the DataFrame
# The target values are:
# 0 = Setosa
# 1 = Versicolor
# 2 = Virginica
df["species"] = iris.target

# Display the first five rows of the dataset
print("First five rows:")
print(df.head())


# Display the names of the three flower species
print("\nTarget names:")
print(iris.target_names)


# Separate the input features (X) from the target (y)
# X contains:
# - sepal length
# - sepal width
# - petal length
# - petal width
X = iris.data

# y contains the flower species
y = iris.target

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# Split the dataset into training and testing sets
# 80% of the data will be used for training
# 20% will be used for testing
# stratify=y keeps the class distribution balanced in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create a Logistic Regression model
# multi_class="ovr" means One-vs-Rest strategy
#
# For three classes, the model creates:
# Class 0 vs Rest
# Class 1 vs Rest
# Class 2 vs Rest
# Create a Logistic Regression model as the base classifier
base_model = LogisticRegression(max_iter=200)

# Wrap the Logistic Regression model inside One-vs-Rest
# For 3 classes, this creates 3 binary classifiers:
# Setosa vs Rest
# Versicolor vs Rest
# Virginica vs Rest
model = OneVsRestClassifier(base_model)

print("\nModel created successfully!")


# Train the model using the training data
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Predict the flower species for the unseen test data
y_pred = model.predict(X_test)

print("\nPredicted classes:")
print(y_pred)


# Convert numeric predictions into actual flower names
predicted_species = iris.target_names[y_pred]

print("\nPredicted species:")
print(predicted_species)


# Calculate the probability of each class
# Each row contains probabilities for:
# Setosa, Versicolor, Virginica
probabilities = model.predict_proba(X_test)

print("\nPredicted probabilities:")
print(probabilities)


# Evaluate the model using accuracy
# Accuracy tells us the proportion of correctly classified flowers
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# Create a confusion matrix
# Rows represent actual classes
# Columns represent predicted classes
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)