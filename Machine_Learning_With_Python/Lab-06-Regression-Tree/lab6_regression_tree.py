# Import Pandas for loading and working with the dataset
import pandas as pd


# Give the exact path to our CSV file
# The CSV is inside the Lab-06-Regression-Tree folder
file_path = r"Machine_Learning_With_Python\Lab-06-Regression-Tree\taxi_tips.csv"


# Load the CSV dataset into a Pandas DataFrame
df = pd.read_csv(file_path)


# Display the first five rows
print("First five rows:")
print(df.head())


# Display the number of rows and columns
print("\nDataset shape:")
print(df.shape)


# Display the column names
print("\nColumn names:")
print(df.columns.tolist())


# Now lets separate the features and the target from out dataset.

# Separate the input features (X) from the target variable (y)

# X contains the information we will use to predict the taxi tip
# We remove tip_amount because that is what we want the model to predict
X = df.drop("tip_amount", axis=1)

# y contains the actual tip amount that the model will learn to predict
y = df["tip_amount"]


# Display the feature data
print("\nFeatures (X):")
print(X.head())


# Display the target values
print("\nTarget (y):")
print(y.head())


# Display the shapes of X and y
print("\nX shape:", X.shape)
print("y shape:", y.shape)


#Now commes the train and test split !

# Import train_test_split to divide the dataset into training and testing sets
from sklearn.model_selection import train_test_split


# Split the data into training and testing sets
# 80% of the data will be used to train the model
# 20% will be used to test the model on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Display the shapes of the resulting datasets
print("\nTraining data:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nTesting data:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Import DecisionTreeRegressor to create a regression tree model
from sklearn.tree import DecisionTreeRegressor


# Create the Regression Tree
# max_depth controls how deep the tree is allowed to grow
# random_state makes our results reproducible
model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)


# Display a confirmation message
print("\nRegression Tree model created successfully!")


# Train the Regression Tree using the training data
# The model learns relationships between the taxi features
# and the tip amounts
model.fit(X_train, y_train)


# Display a confirmation message
print("Regression Tree model trained successfully!")

# Now prediction time !

# Use the trained Regression Tree to predict tip amounts
# for the unseen testing data
y_pred = model.predict(X_test)


# Display the predicted tip amounts
print("\nPredicted tip amounts:")
print(y_pred)


# Display the actual tip amounts from the testing data
print("\nActual tip amounts:")
print(y_test.values)


# Create a table to compare actual and predicted values
results = pd.DataFrame({
    "Actual Tip": y_test.values,
    "Predicted Tip": y_pred
})


# Display the comparison table
print("\nActual vs Predicted:")
print(results)

#Now lets calculate MSE 

# Import mean_squared_error to evaluate the regression model
from sklearn.metrics import mean_squared_error


# Calculate the Mean Squared Error (MSE)
# MSE measures the average squared difference between
# the actual values and the predicted values
mse = mean_squared_error(y_test, y_pred)


# Display the MSE
print("\nMean Squared Error (MSE):")
print(mse)


#Now lets calculate R²

# Import r2_score to evaluate how well the model explains
# the variation in the target values
from sklearn.metrics import r2_score


# Calculate the R² score using the actual and predicted tip amounts
r2 = r2_score(y_test, y_pred)


# Display the R² score
print("\nR² Score:")
print(r2)

#Lets visualize the regression tree now !

# Import matplotlib for displaying the tree
import matplotlib.pyplot as plt

# Import plot_tree to visualize the Decision Tree
from sklearn.tree import plot_tree


# Create a large figure so the tree is easy to read
plt.figure(figsize=(20, 10))


# Draw the Regression Tree
# feature_names shows the names of our input features
# filled=True adds shading to the tree nodes
# rounded=True makes the nodes easier to read
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=9
)


# Add a title to the visualization
plt.title("Regression Tree for Taxi Tip Prediction")


# Adjust spacing so the tree fits properly
plt.tight_layout()


# Display the tree
plt.show()

# Lets check the feature importance now !

# Get the importance of each feature from the trained Regression Tree
# Feature importance tells us how much each feature contributed
# to the tree's prediction decisions
feature_importance = model.feature_importances_


# Create a DataFrame to display feature names with their importance values
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
})


# Sort the features from most important to least important
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# Display the feature importance results
print("\nFeature Importance:")
print(importance_df)


#A hypothetical new trip just to test the model prediction quality.

# Create a new taxi trip that the model has never seen before
# The values represent:
# distance = 6 km
# trip duration = 25 minutes
# passengers = 2
# fare amount = 20
new_trip = pd.DataFrame({
    "distance_km": [6.0],
    "trip_duration_min": [25],
    "passengers": [2],
    "fare_amount": [20.0]
})


# Use the trained Regression Tree to predict the tip
new_tip_prediction = model.predict(new_trip)


# Display the predicted tip amount
print("\nNew Taxi Trip:")
print(new_trip)

print("\nPredicted Tip for New Trip:")
print(new_tip_prediction[0])