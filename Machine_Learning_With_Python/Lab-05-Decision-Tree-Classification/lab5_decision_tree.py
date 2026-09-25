# Import Pandas for working with the dataset
import pandas as pd

# URL of the drug200.csv dataset used in the IBM lab
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"

# Load the dataset directly from the URL
df = pd.read_csv(url)

# Display the first five rows
print("First five rows:")
print(df.head())

# Display the shape of the dataset
print("\nDataset shape:")
print(df.shape)

# Display the column names
print("\nColumn names:")
print(df.columns)

# Display the unique values in each categorical column
# This helps us understand what categories are present
print("\nUnique values in Sex:")
print(df["Sex"].unique())

print("\nUnique values in BP:")
print(df["BP"].unique())

print("\nUnique values in Cholesterol:")
print(df["Cholesterol"].unique())

print("\nUnique values in Drug:")
print(df["Drug"].unique())

# Import LabelEncoder to convert categorical text values into numbers
from sklearn.preprocessing import LabelEncoder


# Create a LabelEncoder object
label_encoder = LabelEncoder()


# Convert Sex values:
# F and M will be converted into numerical values
df["Sex"] = label_encoder.fit_transform(df["Sex"])

# Convert BP values:
# HIGH, LOW, NORMAL will be converted into numerical values
df["BP"] = label_encoder.fit_transform(df["BP"])

# Convert Cholesterol values:
# HIGH and NORMAL will be converted into numerical values
df["Cholesterol"] = label_encoder.fit_transform(df["Cholesterol"])

# Convert Drug values:
# drugA, drugB, drugC, drugX, drugY will be converted into numerical classes
df["Drug"] = label_encoder.fit_transform(df["Drug"])


# Display the dataset after encoding
print("\nDataset after encoding:")
print(df.head())

# Separate the input features (X) from the target variable (y)
# X contains the patient health information used for prediction
X = df.drop("Drug", axis=1)

# y contains the drug class that we want the model to predict
y = df["Drug"]


# Import train_test_split to divide the data into training and testing sets
from sklearn.model_selection import train_test_split


# Split the dataset into training and testing sets
# 80% of the data will be used for training
# 20% will be used for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Display the shapes of the training and testing data
print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)

# Import DecisionTreeClassifier to create the decision tree model
from sklearn.tree import DecisionTreeClassifier


# Create a Decision Tree classifier
# criterion="entropy" means the tree will use entropy/information gain
# to decide the best feature for each split
model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

print("\nDecision Tree model created successfully!")


# Train the decision tree using the training data
# The model learns which patient characteristics are associated
# with each drug class
model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")


# Use the trained Decision Tree to predict drug classes
# for the unseen testing data
y_pred = model.predict(X_test)


# Display the predicted drug classes
print("\nPredicted drug classes:")
print(y_pred)


# Display the actual drug classes for comparison
print("\nActual drug classes:")
print(y_test.values)


#Calculate the accuracy now !

# Import accuracy_score to measure how many predictions were correct
from sklearn.metrics import accuracy_score


# Calculate the accuracy of the Decision Tree model
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:")
print(accuracy)

#Now lets make confusion metrics for final Evaluation.

# Import confusion_matrix to see correct and incorrect predictions
from sklearn.metrics import confusion_matrix


# Create the confusion matrix using actual and predicted values
cm = confusion_matrix(y_test, y_pred)


# Display the confusion matrix
print("\nConfusion Matrix:")
print(cm)

#Now lets visualize the decision tree now !

# Import the functions needed to visualize the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# Create a large figure so the tree has enough space
plt.figure(figsize=(24, 12))


# Display only the first 3 levels of the tree
# This makes the main decision structure easier to understand
plot_tree(
    model,
    max_depth=3,
    feature_names=X.columns,
    class_names=["drugA", "drugB", "drugC", "drugX", "drugY"],
    filled=True,
    rounded=True,
    fontsize=9
)


# Add a title
plt.title("Decision Tree for Drug Classification (First 3 Levels)")


# Adjust spacing so that the tree does not get cut off
plt.tight_layout()


# Display the tree
plt.show()