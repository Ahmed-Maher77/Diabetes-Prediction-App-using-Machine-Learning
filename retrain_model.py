#!/usr/bin/env python3
"""
Script to retrain the diabetes prediction model with current scikit-learn version
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import sklearn

print(f"Using scikit-learn version: {sklearn.__version__}")

# Load the diabetes dataset (you'll need to provide your training data)
# This is a placeholder - replace with your actual training data loading
def load_training_data():
    """
    Load your diabetes training dataset here
    Replace this with your actual data loading code
    """
    # Example structure - replace with your actual data
    # You can load from CSV, database, etc.
    data = pd.read_csv('diabetes.csv')  # Replace with your data file
    return data

def train_model():
    """Train a new RandomForest model with current scikit-learn version"""
    try:
        # Load training data
        print("Loading training data...")
        data = load_training_data()
        
        # Prepare features and target
        X = data.drop('Outcome', axis=1)  # Adjust column name as needed
        y = data['Outcome']  # Adjust column name as needed
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        print("Training RandomForest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        print("Saving model...")
        with open('Diabetes-Prediction-ML-Model.sav', 'wb') as f:
            pickle.dump(model, f)
        
        print("✅ Model trained and saved successfully!")
        print(f"Model saved as: Diabetes-Prediction-ML-Model.sav")
        
        return model
        
    except FileNotFoundError:
        print("❌ Training data file not found. Please provide your diabetes dataset.")
        print("Expected file: diabetes.csv (or modify the load_training_data() function)")
        return None
    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        return None

if __name__ == "__main__":
    print("🔄 Retraining Diabetes Prediction Model")
    print("=" * 50)
    model = train_model()
    
    if model:
        print("\n✅ Model retraining completed successfully!")
        print("You can now deploy this model to Streamlit without version conflicts.")
    else:
        print("\n❌ Model retraining failed. Please check your training data.") 