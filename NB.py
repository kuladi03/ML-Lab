# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("your_dataset.csv")  # Replace "your_dataset.csv" with the path to your dataset file

# Define categories for sales
# You can define your own categories based on your data distribution
# For example, you can use bins or custom categories
bins = [0, 5, 10, 15, 20, 25, 30]  # Define your own bins here
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme']

# Convert sales into categorical variable
data['Sales_Category'] = pd.cut(data['Sales'], bins=bins, labels=labels)

# Separate features (X) and target variable (y)
X = data.drop(['Sales', 'Sales_Category'], axis=1)  # Features
y = data['Sales_Category']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Categorical Naive Bayes classifier
nb_classifier = CategoricalNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Predict the target variable on the testing set
y_pred = nb_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
