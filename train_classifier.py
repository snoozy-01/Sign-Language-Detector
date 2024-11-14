import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load processed data
data_dict = pickle.load(open('./processed_data.pickle', 'rb'))

# Use the correct keys to access features and labels
data = np.asarray(data_dict['features'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate model performance
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
