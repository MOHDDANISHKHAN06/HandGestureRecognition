import json_Reader 
import Feature_Extraction
import Feature_Combination
import Normalise
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = {
    'k-NN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=5000,random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Function to pad sequences to the same length
def pad_sequences(sequences):
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = np.zeros((len(sequences), max_len))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences

gesture_data = json_Reader.readJson('/Users/mohddanishkhan/Downloads/VolunterWork/json/*-Annotated.json') 
feature_data = Feature_Extraction.extractFeatures(gesture_data)
# print(feature_data)
feature_combinations = Feature_Combination.combinations(feature_data)
#print(feature_combinations)

data_labels = Normalise.norm(feature_combinations)
print(len(data_labels))

separated_data_non_normalized = data_labels[0]
separated_data_normalized = data_labels[1]
labels = data_labels[2]

# Converting to NumPy arrays for easier handling
X_non_normalized = np.array(separated_data_non_normalized, dtype=object)
X_normalized = np.array(separated_data_normalized, dtype=object)
y = np.array(labels)

X_train_non_normalized, X_test_non_normalized, y_train_non_normalized, y_test_non_normalized = train_test_split(X_non_normalized, y, test_size=0.2, random_state=42)
X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Pad the non-normalized and normalized numerical features
X_train_non_normalized_padded = pad_sequences([x[1:] for x in X_train_non_normalized])
X_test_non_normalized_padded = pad_sequences([x[1:] for x in X_test_non_normalized])
X_train_normalized_padded = pad_sequences([x[1:] for x in X_train_normalized])
X_test_normalized_padded = pad_sequences([x[1:] for x in X_test_normalized])

performance_metrics = {}

# Train and evaluate each model on both non-normalized and normalized data
for model_name, model in models.items():
    # Metrics for non-normalized data
    model.fit(X_train_non_normalized_padded, y_train_non_normalized)
    y_pred_non_normalized = model.predict(X_test_non_normalized_padded)
    accuracy_non_normalized = accuracy_score(y_test_non_normalized, y_pred_non_normalized)
    precision_non_normalized = precision_score(y_test_non_normalized, y_pred_non_normalized, average='weighted',zero_division=1)
    recall_non_normalized = recall_score(y_test_non_normalized, y_pred_non_normalized, average='weighted',zero_division=1)
    f1_non_normalized = f1_score(y_test_non_normalized, y_pred_non_normalized, average='weighted',zero_division=1)
    
    # Metrics for normalized data
    model.fit(X_train_normalized_padded, y_train_normalized)
    y_pred_normalized = model.predict(X_test_normalized_padded)
    accuracy_normalized = accuracy_score(y_test_normalized, y_pred_normalized)
    precision_normalized = precision_score(y_test_normalized, y_pred_normalized, average='weighted',zero_division=1)
    recall_normalized = recall_score(y_test_normalized, y_pred_normalized, average='weighted',zero_division=1)
    f1_normalized = f1_score(y_test_normalized, y_pred_normalized, average='weighted',zero_division=1)
    
    # Store the metrics
    performance_metrics[model_name] = {
        'Non-Normalized': {
            'Accuracy': accuracy_non_normalized,
            'Precision': precision_non_normalized,
            'Recall': recall_non_normalized,
            'F1 Score': f1_non_normalized
        },
        'Normalized': {
            'Accuracy': accuracy_normalized,
            'Precision': precision_normalized,
            'Recall': recall_normalized,
            'F1 Score': f1_normalized
        }
    }

# Convert the performance metrics dictionary to a Pandas DataFrame for better visualization
performance_df = pd.DataFrame.from_dict({(i, j): performance_metrics[i][j] 
                                         for i in performance_metrics.keys() 
                                         for j in performance_metrics[i].keys()}, 
                                       orient='index')

# Convert the metrics to %age
performance_df *= 100
performance_df = performance_df.round(2)
print(performance_df)