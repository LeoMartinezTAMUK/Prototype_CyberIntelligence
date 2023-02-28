# Final Project Prototype/Template Cyber Intelligence CSEN 4370/5303
# Leo Martinez III, Lidia A. Morales, Babatunde T. Arowolo

# Import the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# Program was created in Spyder 5.2.2 Anaconda with Python 3.9

#%%
# Load the dataset
dataset = pd.read_csv('network_traffic.csv')
# (May need to change the pathway of .csv file above) the dataset used is currently only a sample one, not real data yet.

#%%
# Split the data into features and labels
X = dataset.iloc[:, 4:-1].values
# Independent Variables(X) Source IP, Destination IP, Source Port, and Destination Port are not yet implemented
y = dataset.iloc[:, -1].values
# Dependent Variable(y) is "Mal = 1 Ben = 0"

#%%
# Encoding Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
# Use these methods to convert categorical data such a protocal into data we can use in the ML algorithm

#%%,
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 80% of data will be the training set and 20% will be the testing set


#%%
# Train the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
# Other ML algorithms can be utilized later on to test for best performance

#%%
# Make predictions on the testing set
y_pred = classifier.predict(X_test)

#%%
# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion matrix:", confusion_mat)
