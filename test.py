# Import necessary libraries
import numpy as np
from sklearn.metrics import accuracy_score
import joblib

# Load the preprocessed test data and labels
x_test_path = 'models/X_test.npy'
y_test_path = 'models/y_test.npy'

try:
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    print(f"Loaded test data from {x_test_path} with shape: {X_test.shape}")
    print(f"Loaded test labels from {y_test_path} with shape: {y_test.shape}")
except FileNotFoundError as e:
    print(f"Error: Test file not found - {e}. Ensure training was completed successfully.")
    exit()

# List of tuned model filenames
model_files = {
    'Logistic Regression': 'models/Logistic_Regression_tuned.pkl',
    'Decision Tree': 'models/Decision_Tree_tuned.pkl',
    'Random Forest': 'models/Random_Forest_tuned.pkl',
    'SVM': 'models/SVM_tuned.pkl',
    'KNN': 'models/KNN_tuned.pkl',
    'XGBoost': 'models/XGBoost_tuned.pkl',
    'MLPClassifier': 'models/MLPClassifier_tuned.pkl'
}

# Load and test each tuned model
print("\nPredictions with Preprocessed Test Data from X_test.npy and y_test.npy using Tuned Models:")
for name, filepath in model_files.items():
    try:
        # Load the tuned model
        model = joblib.load(filepath)
        # Predict on test data
        y_pred = model.predict(X_test)
        # Print predictions (limited to first 10 for brevity)
        print(f'{name}: Predictions (first 10) = {y_pred[:10]} (1=Positive, 0=Negative)')
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{name}: Accuracy = {accuracy:.4f}')
        print("---")
    except FileNotFoundError:
        print(f"Error: Model file {filepath} not found. Please ensure the tuned model was saved correctly during training.")
        print("---")