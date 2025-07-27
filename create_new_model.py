#!/usr/bin/env python3
"""
Script to create a new diabetes prediction model with current scikit-learn version
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import sklearn

print(f"Using scikit-learn version: {sklearn.__version__}")

def create_sample_data():
    """Create sample diabetes data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic diabetes data
    pregnancies = np.random.randint(0, 18, n_samples)
    glucose = np.random.normal(120, 30, n_samples)
    glucose = np.clip(glucose, 0, 200)
    blood_pressure = np.random.normal(70, 12, n_samples)
    blood_pressure = np.clip(blood_pressure, 0, 122)
    skin_thickness = np.random.normal(20, 15, n_samples)
    skin_thickness = np.clip(skin_thickness, 0, 99)
    insulin = np.random.normal(80, 80, n_samples)
    insulin = np.clip(insulin, 0, 846)
    bmi = np.random.normal(32, 6.9, n_samples)
    bmi = np.clip(bmi, 0, 67.1)
    diabetes_pedigree = np.random.normal(0.5, 0.3, n_samples)
    diabetes_pedigree = np.clip(diabetes_pedigree, 0.078, 2.42)
    age = np.random.randint(21, 81, n_samples)
    
    # Create features
    features = np.column_stack([
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ])
    
    # Create target (diabetes outcome) based on realistic patterns
    # Higher glucose, age, and BMI increase diabetes risk
    diabetes_risk = (
        glucose * 0.01 + 
        age * 0.02 + 
        bmi * 0.03 + 
        diabetes_pedigree * 0.5 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    outcome = (diabetes_risk > 0.5).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame(features, columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    data['Outcome'] = outcome
    
    return data

def train_new_model():
    """Train a new RandomForest model with current scikit-learn version"""
    try:
        print("Creating sample diabetes dataset...")
        data = create_sample_data()
        
        print(f"Dataset shape: {data.shape}")
        print(f"Diabetes cases: {data['Outcome'].sum()} out of {len(data)}")
        
        # Prepare features and target
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
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
        print(f"Model type: {type(model)}")
        
        # Test the model
        test_sample = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
        prediction = model.predict(test_sample)
        print(f"Test prediction: {prediction[0]}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        return None

if __name__ == "__main__":
    print("🔄 Creating New Diabetes Prediction Model")
    print("=" * 50)
    model = train_new_model()
    
    if model:
        print("\n✅ Model creation completed successfully!")
        print("You can now deploy this model to Streamlit without version conflicts.")
        print("Note: This model uses synthetic data. For production, use your actual diabetes dataset.")
    else:
        print("\n❌ Model creation failed.") 