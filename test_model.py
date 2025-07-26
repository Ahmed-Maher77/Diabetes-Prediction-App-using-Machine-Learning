import pickle
import os
import sys

def test_model_loading():
    """Test if the model file can be loaded successfully"""
    try:
        # Check if file exists
        model_file = 'Diabetes-Prediction-ML-Model.sav'
        if not os.path.exists(model_file):
            print(f"ERROR: Model file '{model_file}' not found!")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in directory: {os.listdir('.')}")
            return False
        
        # Try to load the model
        print(f"Attempting to load model from: {model_file}")
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        print("SUCCESS: Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Test if model has predict method
        if hasattr(model, 'predict'):
            print("SUCCESS: Model has predict method")
        else:
            print("WARNING: Model does not have predict method")
        
        return True
        
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        print(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    test_model_loading() 