
# ============================================================
# Lab 7 - Credit Card Fraud Classification
# Models: Decision Tree and Support Vector Machine (SVM)
# ============================================================


# ------------------------------------------------------------
# 1. Import Required Libraries
# ------------------------------------------------------------

# Import Pandas for loading and working with the dataset
import pandas as pd

# Import train_test_split to divide the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import DecisionTreeClassifier for classification
from sklearn.tree import DecisionTreeClassifier

# Import SVC (Support Vector Classifier) for SVM classification
from sklearn.svm import SVC

# Import evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix


# ------------------------------------------------------------
# 2. Load the Dataset
# ------------------------------------------------------------

# Give the path to our CSV file
# The CSV is inside the same Lab-07 folder as this Python file
file_path = r"Machine_Learning_With_Python\Lab-07-Credit-Card-Fraud-Classification\transactions.csv"


# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)


# Display the first five transactions
print("First five transactions:")
print(df.head())


# Display the number of rows and columns
print("\nDataset shape:")
print(df.shape)


# Display the column names
print("\nColumn names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 3. Separate Features (X) and Target (y)
# ------------------------------------------------------------

# Separate the input features (X) from the target variable (y)

# X contains the information we will use to predict
# whether a transaction is fraudulent or legitimate
X = df.drop("fraud", axis=1)


# y contains the actual fraud labels
# 0 = legitimate transaction
# 1 = fraudulent transaction
y = df["fraud"]


# Display the input features
print("\nFeatures (X):")
print(X.head())


# Display the target values
print("\nTarget (y):")
print(y.head())


# Display the shapes of X and y
print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ------------------------------------------------------------
# 4. Split Data into Training and Testing Sets
# ------------------------------------------------------------

# Split the data into training and testing sets
# 80% of the transactions will be used for training
# 20% of the transactions will be used for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Display the shapes of the training and testing data
print("\nTraining data:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nTesting data:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ------------------------------------------------------------
# 5. Decision Tree Classification
# ------------------------------------------------------------

# Create the Decision Tree model
# The model will learn rules from the training data
model = DecisionTreeClassifier(
    random_state=42
)


# Train the model using the training data
# X_train = transaction features
# y_train = actual fraud labels
model.fit(X_train, y_train)


# Confirm that the model has been trained
print("\nDecision Tree model trained successfully!")


# ------------------------------------------------------------
# 6. Decision Tree Predictions
# ------------------------------------------------------------

# Use the trained Decision Tree to predict fraud labels
# The model has never seen X_test during training
y_pred = model.predict(X_test)


# Display the predicted fraud labels
print("\nPredicted fraud labels:")
print(y_pred)


# Display the actual fraud labels for comparison
print("\nActual fraud labels:")
print(y_test.values)


# ------------------------------------------------------------
# 7. Evaluate the Decision Tree
# ------------------------------------------------------------

# Calculate the accuracy of the Decision Tree
# Accuracy = correctly predicted transactions / total test transactions
accuracy = accuracy_score(y_test, y_pred)


# Create a confusion matrix
# Rows = actual labels
# Columns = predicted labels
conf_matrix = confusion_matrix(y_test, y_pred)


# Display the evaluation results
print("\nDecision Tree Accuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(conf_matrix)


# ------------------------------------------------------------
# 8. Support Vector Machine (SVM) Classification
# ------------------------------------------------------------

# Create the SVM model
# kernel="rbf" allows the model to learn non-linear boundaries
svm_model = SVC(
    kernel="rbf",
    random_state=42
)


# Train the SVM using the training data
svm_model.fit(X_train, y_train)


# Confirm that the SVM has been trained
print("\nSVM model trained successfully!")


# ------------------------------------------------------------
# 9. SVM Predictions
# ------------------------------------------------------------

# Use the trained SVM model to predict fraud labels
y_pred_svm = svm_model.predict(X_test)


# Display the SVM predictions
print("\nSVM Predicted fraud labels:")
print(y_pred_svm)


# Display the actual labels for comparison
print("\nActual fraud labels:")
print(y_test.values)


# ------------------------------------------------------------
# 10. Evaluate the SVM
# ------------------------------------------------------------

# Calculate the accuracy of the SVM model
svm_accuracy = accuracy_score(y_test, y_pred_svm)


# Create a confusion matrix for the SVM predictions
svm_conf_matrix = confusion_matrix(y_test, y_pred_svm)


# Display the SVM evaluation results
print("\nSVM Accuracy:")
print(svm_accuracy)

print("\nSVM Confusion Matrix:")
print(svm_conf_matrix)


# ------------------------------------------------------------
# 11. Predict a New Transaction
# ------------------------------------------------------------

# Create a new transaction that the models have not seen before
new_transaction = pd.DataFrame({
    "amount": [500],
    "transaction_hour": [2],
    "international": [1],
    "previous_fraud": [1]
})


# Use the Decision Tree to predict the new transaction
new_prediction_tree = model.predict(new_transaction)


# Use the SVM to predict the new transaction
new_prediction_svm = svm_model.predict(new_transaction)


# Display the new transaction
print("\nNew Transaction:")
print(new_transaction)


# Display the Decision Tree prediction
print("\nDecision Tree Prediction:")
print(new_prediction_tree[0])


# Display the SVM prediction
print("\nSVM Prediction:")
print(new_prediction_svm[0])
