import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the spectroscopic data
spectra = np.loadtxt('spectra_data.txt', delimiter=',')

# Load the corresponding object names
object_names = np.loadtxt('object_names.txt', dtype=str)

# Load the chemical element labels
chemical_elements = np.loadtxt('chemical_elements.txt', dtype=str)

# Split the data into training and testing sets
train_data = spectra[:800]
train_labels = object_names[:800]
test_data = spectra[800:]
test_labels = object_names[800:]

# Train a random forest classifier
classifier = RandomForestClassifier()
classifier.fit(train_data, train_labels)

# Predict the object names for the test data
predicted_labels = classifier.predict(test_data)

# Create a markdown table to display the results
result_table = "| Object Name | "
for element in chemical_elements:
    result_table += element + " | "
result_table += "\n| --- |"
for _ in range(len(chemical_elements)):
    result_table += " --- |"
result_table += "\n"

for i in range(len(test_labels)):
    result_table += "| " + test_labels[i] + " | "
    for j in range(len(chemical_elements)):
        result_table += predicted_labels[i][j] + " | "
    result_table += "\n"

print(result_table)
