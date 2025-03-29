#Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Create a directory to store models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the dataset
data = pd.read_csv(r'diabetes_data_upload.csv')

# Map string categorical values to integers
data['class'] = data['class'].map({'Positive': 1, 'Negative': 0})  # Map target variable
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})  # Assuming 'Gender' in your CSV
yes_no_cols = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 
               'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 
               'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']
for col in yes_no_cols:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

# Separate features and target
X = data.drop('class', axis=1)
y = data['class']

# Define numerical and categorical columns
numerical_cols = ['Age']
categorical_cols = [col for col in X.columns if col not in numerical_cols]

# Handle missing values
numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numerical_cols),
        ('cat', categorical_imputer, categorical_cols)
    ]
)
X_imputed = preprocessor.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=numerical_cols + categorical_cols)

# Convert categorical features to integer type
for col in categorical_cols:
    X_imputed_df[col] = X_imputed_df[col].astype(int)

# Check for remaining missing values
print("Remaining missing values:", X_imputed_df.isnull().sum().sum())

# Check for outliers in 'Age' and cap them
sns.boxplot(x=X_imputed_df['Age'])
plt.title("Boxplot of Age")
plt.show()
age_99 = X_imputed_df['Age'].quantile(0.99)
X_imputed_df['Age'] = np.where(X_imputed_df['Age'] > age_99, age_99, X_imputed_df['Age'])

# Scale numerical features (Age), pass through categorical features
scaler = StandardScaler()
preprocessor_scale = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_cols),
        ('cat', 'passthrough', categorical_cols)
    ]
)
X_scaled = preprocessor_scale.fit_transform(X_imputed_df)

# Save the preprocessing pipeline
joblib.dump(preprocessor, 'models/preprocessor_impute.pkl')
joblib.dump(preprocessor_scale, 'models/preprocessor_scale.pkl')

# Check class distribution and handle imbalance if necessary
print("\nClass Distribution:")
print(y.value_counts(normalize=True))
if y.value_counts(normalize=True).min() < 0.4:
    smote = SMOTE(random_state=42)
    X_scaled, y = smote.fit_resample(X_scaled, y)
    print("Applied SMOTE to handle class imbalance.")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save the test set for later use
np.save('models/X_test.npy', X_test)
np.save('models/y_test.npy', y_test)

# Define models and their hyperparameter grids
models = {
    'Logistic Regression': {
        'model': LogisticRegression(),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    },
    'SVM': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    },
    'MLPClassifier': {
        'model': MLPClassifier(max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.01]
        }
    }
}

# Train, tune, and save each model
print("\nTraining, Tuning, and Saving Models:")
for name, model_dict in models.items():
    print(f"\nTuning {name}...")
    grid_search = GridSearchCV(
        estimator=model_dict['model'],
        param_grid=model_dict['params'],
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1  # Use all available CPU cores
    )
    grid_search.fit(X_train, y_train)
    
    # Best model after tuning
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f'{name} Best Params: {grid_search.best_params_}')
    print(f'{name} Accuracy: {accuracy:.4f}')
    
    # Save the tuned model to disk
    filename = f'models/{name.replace(" ", "_")}_tuned.pkl'
    joblib.dump(best_model, filename)
    print(f'Saved tuned {name} to {filename}')